const std = @import("std");
const print = std.debug.print;

const ConstraintType = enum { LessOrEqual, GreaterOrEqual, Equal };
const Constraint = struct { coefficients: []f64, bound: f64, type: ConstraintType };
const Bound = struct { lower: f64, upper: f64 };

const Solver = struct {
    allocator: std.mem.Allocator,
    objective: std.ArrayList(f64) = .empty,
    constraints: std.ArrayList(Constraint) = .empty,
    variable_bounds: std.ArrayList(Bound) = .empty,
    is_integer: std.ArrayList(bool) = .empty,

    pub fn init(allocator: std.mem.Allocator) Solver {
        return .{
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Solver) void {
        self.objective.deinit(self.allocator);
        for (self.constraints.items) |c| self.allocator.free(c.coefficients);
        self.constraints.deinit(self.allocator);
        self.variable_bounds.deinit(self.allocator);
        self.is_integer.deinit(self.allocator);
    }

    pub fn addVariable(self: *Solver, obj_coeff: f64, lower: f64, upper: f64, is_int: bool) !usize {
        try self.objective.append(self.allocator, obj_coeff);
        try self.variable_bounds.append(self.allocator, .{ .lower = lower, .upper = upper });
        try self.is_integer.append(self.allocator, is_int);
        return self.objective.items.len - 1;
    }

    pub fn addConstraint(self: *Solver, coeffs: []const f64, bound: f64, typ: ConstraintType) !void {
        const owned_coeffs = try self.allocator.dupe(f64, coeffs);
        try self.constraints.append(self.allocator, .{ .coefficients = owned_coeffs, .bound = bound, .type = typ });
    }

    fn printMat(self: *Solver, mat: [][]f64) void {
        _ = self;
        print("mat:\n", .{});
        for (mat) |row| {
            print("{any}\n", .{row});
        }
    }

    pub fn solve(self: *Solver) ![]f64 {
        // set the tableau
        const constraints_len = self.constraints.items.len;
        const variables_len = self.variable_bounds.items.len;

        const row_count = constraints_len;
        const col_count = variables_len + constraints_len;
        var objective_row = try self.allocator.alloc(f64, col_count);
        var rhs = try self.allocator.alloc(f64, constraints_len);
        // this is used as tableau
        // row major
        var mat = try self.allocator.alloc([]f64, row_count);

        // set mat
        for (mat, 0..) |*row, contraint_id| {
            row.* = try self.allocator.alloc(f64, col_count);

            for (0..variables_len) |idx| {
                row.*[idx] = self.constraints.items[contraint_id].coefficients[idx];
            }
            for (0..constraints_len) |idx| {
                row.*[variables_len + idx] = 0;
            }
            row.*[variables_len + contraint_id] = 1;
        }

        // set objective row
        for (0..variables_len) |idx| {
            objective_row[idx] = -self.objective.items[idx];
        }

        // set constraints term color
        for (0..constraints_len) |idx| {
            rhs[idx] = self.constraints.items[idx].bound;
        }

        // process tableau
        while (true) {
            const status = try self.iterate(&objective_row, &mat, &rhs);

            switch (status) {
                .Optimal => break,
                .Unbounded => @panic("unbounded"),
                .Pivot => continue,
            }
        }

        return try self.extractSolution(mat, rhs);
    }

    const IterateResult = enum { Optimal, Unbounded, Pivot };

    fn iterate(
        self: *Solver,
        objective_row: *[]f64,
        mat: *[][]f64,
        rhs: *[]f64,
    ) !IterateResult {
        _ = self;

        const rows = rhs.len;
        const cols = mat.*[0].len;

        // 1. find entering column
        var most_neg: f64 = 0.0;
        var entering: ?usize = null;
        for (objective_row.*, 0..) |v, c| {
            if (v < most_neg) {
                most_neg = v;
                entering = c;
            }
        }
        if (entering == null) {
            return .Optimal;
        }
        const col = entering.?;

        // 2. find leaving row
        var best_ratio: f64 = 0xffffffff;
        var leaving: ?usize = null;
        for (0..rows) |r| {
            const a = mat.*[r][col];
            if (a > 0) {
                const ratio = rhs.*[r] / a;
                if (ratio < best_ratio) {
                    best_ratio = ratio;
                    leaving = r;
                }
            }
        }
        if (leaving == null) {
            return .Unbounded;
        }
        const row = leaving.?;

        // 3. normalize pivot
        const pivot = mat.*[row][col];
        for (0..cols) |c| {
            mat.*[row][c] /= pivot;
        }
        rhs.*[row] /= pivot;

        // 4. eliminate
        for (0..rows) |r| {
            if (r == row) continue;
            const factor = mat.*[r][col];
            for (0..cols) |c| {
                mat.*[r][c] -= factor * mat.*[row][c];
            }
            rhs.*[r] -= factor * rhs.*[row];
        }
        const obj_factor = objective_row.*[col];
        for (0..cols) |c| {
            objective_row.*[c] -= obj_factor * mat.*[row][c];
        }

        return .Pivot;
    }

    fn extractSolution(
        self: *Solver,
        mat: [][]f64,
        rhs: []f64,
    ) ![]f64 {
        const variables_len = self.variable_bounds.items.len;
        const rows = rhs.len;

        var solution = try self.allocator.alloc(f64, variables_len);

        for (0..variables_len) |i| solution[i] = 0;

        for (0..variables_len) |var_idx| {
            var one_row: ?usize = null;
            var valid_basic = true;

            for (0..rows) |r| {
                const v = mat[r][var_idx];
                if (std.math.approxEqAbs(f64, v, 1.0, 1e-9)) {
                    if (one_row != null) {
                        valid_basic = false;
                        break;
                    }
                    one_row = r;
                } else if (!std.math.approxEqAbs(f64, v, 0.0, 1e-9)) {
                    valid_basic = false;
                    break;
                }
            }

            if (valid_basic and one_row != null) {
                solution[var_idx] = rhs[one_row.?];
            } else {
                solution[var_idx] = 0;
            }
        }

        return solution;
    }
};

// MINIMIZE 5*x_a+4*x_b
// s.t
//    3*x_a + 5*x_b <= 78
//    4*x_a + x_b <= 36
//    x_a, x_b >= 0
test "the youtube test" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var solver = Solver.init(allocator);

    _ = try solver.addVariable(5.0, 0.0, 0xffffffff, false);
    _ = try solver.addVariable(4.0, 0.0, 0xffffffff, false);

    const coeffs1 = [_]f64{ 3.0, 5.0 };
    try solver.addConstraint(&coeffs1, 78.0, .LessOrEqual);

    const coeffs2 = [_]f64{ 4.0, 1.0 };
    try solver.addConstraint(&coeffs2, 36.0, .LessOrEqual);

    const solution = try solver.solve();
    const expected = [_]f64{ 6, 12 };
    try std.testing.expectEqualSlices(f64, solution, &expected);
}

// minimize 1*x_a + 15*x_mul + 2*x_b + 3*x_c
// s.t.
//    x_a + x_mul = 1
//    x_mul - x_b <= 0
//    x_mul - x_c <= 0
//    all x are binary [0, 1]
//
//solution
//   { 1, 0, 0, 0 }
test "the z.ai test" {
    if (true) return;
    const allocator = std.testing.allocator;

    var solver = Solver.init(allocator);

    _ = try solver.addVariable(1.0, 0.0, 1.0, true); // x_a
    _ = try solver.addVariable(15.0, 0.0, 1.0, true); // x_mul
    _ = try solver.addVariable(2.0, 0.0, 1.0, true); // x_b
    _ = try solver.addVariable(3.0, 0.0, 1.0, true); // x_c

    const coeffs1 = [_]f64{ 1.0, 1.0, 0.0, 0.0 };
    try solver.addConstraint(&coeffs1, 1.0, .Equal);

    const coeffs2 = [_]f64{ 0.0, 1.0, -1.0, 0.0 };
    try solver.addConstraint(&coeffs2, 0.0, .LessOrEqual);

    const coeffs3 = [_]f64{ 0.0, 1.0, 0.0, -1.0 };
    try solver.addConstraint(&coeffs3, 0.0, .LessOrEqual);

    const solution = try solver.solve();
    _ = solution;
}
