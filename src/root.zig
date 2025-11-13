const std = @import("std");
const print = std.debug.print;

const ConstraintType = enum { LessOrEqual, GreaterOrEqual, Equal };
const Constraint = struct { coefficients: []f64, bound: f64, type: ConstraintType };
const Bound = struct { lower: f64, upper: f64 };

const Basis = struct {
    basis_col_of_row: []usize,
};

const ColMap = struct {
    n_vars: usize,
    n_slacks: usize,
    n_artifs: usize,
    first_slack_col: usize,
    first_artif_col: usize,
    total_cols: usize,
};

pub const Solver = struct {
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

    const IterateResult = enum { Optimal, Unbounded, Pivot };

    fn iterate(
        self: *Solver,
        objective_row: *[]f64,
        mat: *[][]f64,
        rhs: *[]f64,
        objective_rhs: *f64,
    ) !IterateResult {
        _ = self;

        const rows = rhs.len;
        const cols = mat.*[0].len;

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

        const pivot = mat.*[row][col];
        for (0..cols) |c| {
            mat.*[row][c] /= pivot;
        }
        rhs.*[row] /= pivot;

        for (0..rows) |r| {
            if (r == row) continue;
            const factor = mat.*[r][col];
            if (factor == 0) continue;
            for (0..cols) |c| {
                mat.*[r][c] -= factor * mat.*[row][c];
            }
            rhs.*[r] -= factor * rhs.*[row];
        }

        const obj_factor = objective_row.*[col];
        if (obj_factor != 0) {
            for (0..cols) |c| {
                objective_row.*[c] -= obj_factor * mat.*[row][c];
            }
            objective_rhs.* -= obj_factor * rhs.*[row];
        }

        return .Pivot;
    }

    fn buildPhase1(self: *Solver) !struct {
        mat: [][]f64,
        rhs: []f64,
        obj: []f64,
        obj_rhs: f64,
        basis: Basis,
        col_map: ColMap,
    } {
        const allocator = self.allocator;
        const m = self.constraints.items.len;
        const n = self.variable_bounds.items.len;

        var slack_count: usize = 0;
        var artif_count: usize = 0;
        for (self.constraints.items) |c| {
            switch (c.type) {
                .LessOrEqual => slack_count += 1,
                .GreaterOrEqual => {
                    slack_count += 1;
                    artif_count += 1;
                },
                .Equal => {
                    artif_count += 1;
                },
            }
        }

        const first_slack = n;
        const first_artif = n + slack_count;
        const total_cols = n + slack_count + artif_count;

        var mat = try allocator.alloc([]f64, m);
        var rhs = try allocator.alloc(f64, m);
        var obj = try allocator.alloc(f64, total_cols);
        for (0..total_cols) |j| obj[j] = 0;

        var basis_cols = try allocator.alloc(usize, m);

        var slack_idx: usize = 0;
        var artif_idx: usize = 0;

        for (0..m) |r| {
            mat[r] = try allocator.alloc(f64, total_cols);
            for (0..total_cols) |j| mat[r][j] = 0.0;

            for (0..n) |j| {
                mat[r][j] = self.constraints.items[r].coefficients[j];
            }
            rhs[r] = self.constraints.items[r].bound;

            switch (self.constraints.items[r].type) {
                .LessOrEqual => {
                    const sc = first_slack + slack_idx;
                    mat[r][sc] = 1.0;
                    basis_cols[r] = sc;
                    slack_idx += 1;
                },
                .GreaterOrEqual => {
                    const sc = first_slack + slack_idx;
                    mat[r][sc] = -1.0;
                    slack_idx += 1;

                    const ac = first_artif + artif_idx;
                    mat[r][ac] = 1.0;
                    basis_cols[r] = ac;
                    artif_idx += 1;
                },
                .Equal => {
                    const ac = first_artif + artif_idx;
                    mat[r][ac] = 1.0;
                    basis_cols[r] = ac;
                    artif_idx += 1;
                },
            }
        }

        for (0..artif_count) |k| obj[first_artif + k] = -1.0;

        var obj_rhs: f64 = 0.0;
        for (0..m) |r| {
            const bc = basis_cols[r];
            if (bc >= first_artif and bc < first_artif + artif_count) {
                for (0..total_cols) |j| {
                    obj[j] += mat[r][j];
                }
                obj_rhs += rhs[r];
            }
        }

        return .{
            .mat = mat,
            .rhs = rhs,
            .obj = obj,
            .obj_rhs = obj_rhs,
            .basis = .{ .basis_col_of_row = basis_cols },
            .col_map = .{
                .n_vars = n,
                .n_slacks = slack_count,
                .n_artifs = artif_count,
                .first_slack_col = first_slack,
                .first_artif_col = first_artif,
                .total_cols = total_cols,
            },
        };
    }

    fn pivotOutArtificialAndDrop(
        self: *Solver,
        mat_in: [][]f64,
        rhs_in: []f64,
        basis: *Basis,
        colmap: *const ColMap,
    ) !struct { mat: [][]f64, rhs: []f64, basis: Basis } {
        const allocator = self.allocator;
        const m = rhs_in.len;
        const n_keep = colmap.n_vars + colmap.n_slacks;

        for (0..m) |r| {
            const bc = basis.basis_col_of_row[r];
            if (bc >= colmap.first_artif_col and bc < colmap.first_artif_col + colmap.n_artifs) {
                var enter: ?usize = null;
                for (0..n_keep) |c| {
                    if (!std.math.approxEqAbs(f64, mat_in[r][c], 0.0, 1e-12)) {
                        enter = c;
                        break;
                    }
                }
                if (enter) |ec| {
                    const cols = colmap.total_cols;
                    const pivot = mat_in[r][ec];
                    for (0..cols) |j| {
                        mat_in[r][j] = mat_in[r][j] / pivot;
                    }
                    rhs_in[r] = rhs_in[r] / pivot;
                    for (0..m) |rr| {
                        if (rr == r) continue;
                        const factor = mat_in[rr][ec];
                        if (factor == 0) continue;
                        for (0..cols) |j| {
                            mat_in[rr][j] -= factor * mat_in[r][j];
                        }
                        rhs_in[rr] -= factor * rhs_in[r];
                    }
                    basis.basis_col_of_row[r] = ec;
                }
            }
        }

        var mat = try allocator.alloc([]f64, m);
        for (0..m) |r| {
            mat[r] = try allocator.alloc(f64, n_keep);
            for (0..n_keep) |c| mat[r][c] = mat_in[r][c];
        }
        var rhs = try allocator.alloc(f64, m);
        for (0..m) |r| rhs[r] = rhs_in[r];

        var new_basis = try allocator.alloc(usize, m);
        for (0..m) |r| {
            const bc = basis.basis_col_of_row[r];
            if (bc < n_keep) new_basis[r] = bc else new_basis[r] = std.math.maxInt(usize);
        }

        return .{ .mat = mat, .rhs = rhs, .basis = .{ .basis_col_of_row = new_basis } };
    }

    fn buildPhase2Objective(
        self: *Solver,
        mat: [][]f64,
        rhs: []f64,
        basis: Basis,
        n_keep_cols: usize,
    ) !struct { obj: []f64, obj_rhs: f64 } {
        const allocator = self.allocator;
        const n = self.variable_bounds.items.len;

        var obj = try allocator.alloc(f64, n_keep_cols);
        for (0..n_keep_cols) |j| obj[j] = 0.0;

        for (0..n) |j| obj[j] = -self.objective.items[j];

        var obj_rhs: f64 = 0.0;
        for (0..rhs.len) |r| {
            const bc = basis.basis_col_of_row[r];
            if (bc == std.math.maxInt(usize)) continue;
            var c_b: f64 = 0.0;
            if (bc < n) c_b = self.objective.items[bc] else c_b = 0.0;
            if (c_b != 0.0) {
                for (0..n_keep_cols) |j| {
                    obj[j] += c_b * mat[r][j];
                }
                obj_rhs += c_b * rhs[r];
            }
        }

        return .{ .obj = obj, .obj_rhs = obj_rhs };
    }

    pub fn solve(self: *Solver) ![]f64 {
        var p1 = try self.buildPhase1();
        while (true) {
            const status = try self.iterate(&p1.obj, &p1.mat, &p1.rhs, &p1.obj_rhs);
            if (status == .Optimal) break;
            if (status == .Unbounded) @panic("Phase 1 should not be unbounded");
        }
        if (!std.math.approxEqAbs(f64, p1.obj_rhs, 0.0, 1e-8)) {
            @panic("infeasible");
        }

        var dropped = try self.pivotOutArtificialAndDrop(p1.mat, p1.rhs, &p1.basis, &p1.col_map);
        const n_keep = p1.col_map.n_vars + p1.col_map.n_slacks;
        var p2o = try self.buildPhase2Objective(dropped.mat, dropped.rhs, dropped.basis, n_keep);

        while (true) {
            const status = try self.iterate(&p2o.obj, &dropped.mat, &dropped.rhs, &p2o.obj_rhs);
            switch (status) {
                .Optimal => break,
                .Unbounded => @panic("unbounded"),
                .Pivot => {},
            }
        }

        const variables_len = self.variable_bounds.items.len;
        const rows = dropped.rhs.len;
        var solution = try self.allocator.alloc(f64, variables_len);
        for (0..variables_len) |i| solution[i] = 0.0;

        for (0..variables_len) |var_idx| {
            var one_row: ?usize = null;
            var valid_basic = true;

            for (0..rows) |r| {
                const v = dropped.mat[r][var_idx];
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
                solution[var_idx] = dropped.rhs[one_row.?];
            }
        }

        return solution;
    }
};

// minimize 5*x_a+4*x_b
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
// solution:
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
    const expected = [_]f64{ 1, 0, 0, 0 };
    std.testing.expectEqualSlices(f64, solution, expected);
}
