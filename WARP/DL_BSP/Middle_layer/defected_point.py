import ast
import re
import os
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SOLUTION_RE = re.compile(r"^Solution\s+(\d+)\s*$")
X_LINE_RE = re.compile(r"^x:(\d+)\s*\[(.*)\]\s*$")
INPUT_RE = re.compile(r"^x(\d+)\s*=\s*(\[.*\])\s*$")


class DefectPointParser:
    def __init__(self, input_file, bit_width=4):
        self.input_file = input_file
        self.bit_width = bit_width

    def parse(self):
        input_path = Path(self.input_file)
        if not input_path.exists():
            raise FileNotFoundError("file not found: %s" % input_path)

        lines = input_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        solutions = self.parse_solutions(lines)
        solution_inputs = self.parse_solution_inputs(lines)
        incomplete = self.find_incomplete_x(solutions)

        total_solutions = 0
        total_fixed_bits = 0
        line_no = 1
        entries = []

        for solution_id in sorted(incomplete):
            input_map = solution_inputs.get(solution_id, {})
            nonzero_inputs = []
            for x_name_idx in sorted(input_map):
                nonzero_inputs.extend(self.extract_nonzero_entries(input_map[x_name_idx]))

            has_defect = False
            for x_idx in sorted(incomplete[solution_id]):
                vals = incomplete[solution_id][x_idx]
                fixed_bits = self.find_fixed_bits(vals, self.bit_width)
                if not fixed_bits:
                    continue

                has_defect = True

                input_cell = x_idx
                input_val = None
                for item in nonzero_inputs:
                    if item[0] == x_idx:
                        input_val = item[1]
                        break
                if input_val is None and len(nonzero_inputs) == 1:
                    input_cell, input_val = nonzero_inputs[0]
                if input_val is None:
                    input_val = 0

                for bit_pos in sorted(fixed_bits):
                    entries.append((line_no, input_cell, input_val, x_idx, bit_pos))
                    line_no += 1
                    total_fixed_bits += 1

            if has_defect:
                total_solutions += 1

        return entries, total_solutions, total_fixed_bits

    @staticmethod
    def parse_number_list(raw):
        if not raw.strip():
            return []
        return [int(part.strip()) for part in raw.split(",") if part.strip()]

    @staticmethod
    def parse_solutions(lines):
        solutions = {}
        current_solution = None

        for line in lines:
            line = line.strip()
            solution_match = SOLUTION_RE.match(line)
            if solution_match:
                current_solution = int(solution_match.group(1))
                solutions[current_solution] = {}
                continue

            if current_solution is None:
                continue

            x_match = X_LINE_RE.match(line)
            if not x_match:
                continue

            x_index = int(x_match.group(1))
            values = DefectPointParser.parse_number_list(x_match.group(2))
            solutions[current_solution][x_index] = values

        return solutions

    @staticmethod
    def parse_solution_inputs(lines):
        inputs = {}
        current_solution = None

        for line in lines:
            line = line.strip()
            solution_match = SOLUTION_RE.match(line)
            if solution_match:
                current_solution = int(solution_match.group(1))
                inputs[current_solution] = {}
                continue

            if current_solution is None:
                continue

            input_match = INPUT_RE.match(line)
            if not input_match:
                continue

            x_name_idx = int(input_match.group(1))
            raw_list = input_match.group(2)
            try:
                parsed = ast.literal_eval(raw_list)
            except (SyntaxError, ValueError):
                continue

            if isinstance(parsed, list):
                inputs[current_solution][x_name_idx] = parsed

        return inputs

    @staticmethod
    def is_complete_16(values):
        return len(values) == 16 and set(values) == set(range(16))

    @staticmethod
    def find_incomplete_x(solutions):
        result = {}
        for solution_id, x_map in solutions.items():
            incomplete = {x_idx: vals for x_idx, vals in x_map.items() if not DefectPointParser.is_complete_16(vals)}
            if incomplete:
                result[solution_id] = incomplete
        return result

    @staticmethod
    def find_fixed_bits(values, bit_width=4):
        if not values:
            return {}
        fixed = {}
        for pos in range(bit_width):
            shift = bit_width - 1 - pos
            bits = {(v >> shift) & 1 for v in values}
            if len(bits) == 1:
                fixed[pos + 1] = bits.pop()
        return fixed

    @staticmethod
    def extract_nonzero_entries(pairs):
        result = []
        for item in pairs:
            if not isinstance(item, list) or len(item) != 2:
                continue
            idx, val = item
            if isinstance(val, int) and val != 0:
                result.append([idx, val])
        return result


if __name__ == "__main__":
    r = 11
    out_file = os.path.join(BASE_DIR, "result", "defect_points.txt")
    entries, total_solutions, total_fixed_bits = DefectPointParser(
        os.path.join(BASE_DIR, "result", "middle_layer_%d.txt" % r)
    ).parse()
    with open(out_file, "w", encoding="utf-8") as file:
        for line_no, input_cell, input_val, cell, point in entries:
            file.write("%d: input[%d, %d]  cell: %d  defect point: %d\n" % (line_no, input_cell, input_val, cell, point))
        file.write("\ndefect cell num = %d\n" % total_solutions)
        file.write("defect point num = %d\n" % total_fixed_bits)
    print("defect cell num = %d" % total_solutions)
    print("defect point num = %d" % total_fixed_bits)
