#!/usr/bin/env python3

import argparse
import os
import re


def parse_range_or_value(item_str):
    item_str = item_str.strip()
    if '-' in item_str:
        try:
            start, end = map(int, item_str.split('-'))
            return list(range(start, end + 1))
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid range: '{item_str}'")
    else:
        try:
            return [int(item_str)]
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid value: '{item_str}'")


def parse_mixed_list(input_str):
    s = input_str.strip().strip('[]')
    if not s:
        return []
    items = re.split(r'\s*,\s*', s)
    result = []
    for it in items:
        result.extend(parse_range_or_value(it))
    return result


def parse_list_of_mo_lists(input_str):
    s = input_str.strip().strip('[]')
    if not s:
        return []
    groups = re.split(r'\s*:\s*', s)
    result = []
    for g in groups:
        items = re.split(r'\s*,\s*', g)
        sub = []
        for it in items:
            sub.extend(parse_range_or_value(it))
        result.append(sub)
    return result


class ParseMixedListAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        joined = ' '.join(values)
        parsed = parse_mixed_list(joined)
        setattr(namespace, self.dest, parsed)


class ParseMoListListAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        joined = ' '.join(values)
        parsed = parse_list_of_mo_lists(joined)
        setattr(namespace, self.dest, parsed)


def read_alter_swaps(alter_file):
    with open(alter_file, 'r') as f:
        text = f.read()

    alter_match = re.search(r'ALTER\s*=\s*(\d+)\s*;', text, flags=re.IGNORECASE)
    if alter_match is None:
        raise ValueError(f"Cannot find 'ALTER = N;' in file: {alter_file}")
    declared_n = int(alter_match.group(1))

    swaps = [(int(a), int(b)) for a, b in re.findall(r'\b1\s+(\d+)\s+(\d+)\s*;', text)]
    if declared_n != len(swaps):
        raise ValueError(
            f"ALTER count mismatch in {alter_file}: declared {declared_n}, found {len(swaps)} swap entries"
        )
    return swaps


def write_alter_file(target_orbitals, active_orbitals, alter_path):
    target_not_in_active = [t for t in target_orbitals if t not in active_orbitals]
    active_not_in_target = [a for a in active_orbitals if a not in target_orbitals]

    with open(alter_path, 'w') as alterfile:
        line = 'ALTER = ' + str(len(active_not_in_target)) + '; '
        for i in range(len(active_not_in_target)):
            line += '1 ' + str(target_not_in_active[i]) + ' ' + str(active_not_in_target[i]) + '; '
        line += ' * Generated automatically\n'
        alterfile.write(line)


def validate_no_overlap(spaces):
    seen = set()
    for idx, sp in enumerate(spaces):
        overlap = seen.intersection(sp)
        if overlap:
            raise ValueError(
                f"Active-space lists must be disjoint across files. "
                f"Found overlap in list {idx + 1}: {sorted(overlap)}"
            )
        seen.update(sp)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Combine multiple OpenMolcas ALTER files into one ALTER file.\n\n"
            "Each input ALTER file must be paired with the local active space used to create it.\n"
            "The script resolves each fragment's swaps independently, concatenates the resulting\n"
            "target orbital lists, and reconstructs the final ALTER block for the whole molecule."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Example:\n"
            "  python combine_alter_files.py \\\n"
            "    -i ALTER1.txt ALTER2.txt \\\n"
            "    --active_spaces 19-23:30-34 \\\n"
            "    --total_active_space 19-23,30-34 \\\n"
            "    -o ALTER_total.txt"
        )
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        nargs='+',
        metavar='ALTER_FILE',
        help="List of ALTER files to combine (space-separated)."
    )
    parser.add_argument(
        '--active_spaces',
        required=True,
        nargs='+',
        action=ParseMoListListAction,
        metavar='ACTIVE_SPACES',
        help=(
            "Active-space list used for each input ALTER file (1-based indices).\n"
            "Use commas for values, hyphens for ranges, and colons to separate lists.\n"
            "Example: 19-23:30-34"
        )
    )
    parser.add_argument(
        '--total_active_space',
        required=True,
        nargs='+',
        action=ParseMixedListAction,
        metavar='TOTAL_ACTIVE',
        help=(
            "Total active space of the whole molecule (1-based indices).\n"
            "Example: 19-23,30-34"
        )
    )
    parser.add_argument(
        '-o', '--output',
        required=False,
        default='ALTER.txt',
        metavar='OUTPUT_ALTER',
        help="Output ALTER filename/path. Default: ALTER.txt"
    )

    args = parser.parse_args()

    if len(args.input) != len(args.active_spaces):
        raise ValueError("The number of --input files must match the number of lists in --active_spaces")

    for path in args.input:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Input ALTER file not found: {path}")

    validate_no_overlap(args.active_spaces)

    total_active = list(args.total_active_space)
    if len(total_active) != len(set(total_active)):
        raise ValueError("--total_active_space contains duplicate orbitals")

    for idx, local_active in enumerate(args.active_spaces):
        missing = [orb for orb in local_active if orb not in total_active]
        if missing:
            raise ValueError(
                f"Active-space list {idx + 1} contains orbitals not present in --total_active_space: {missing}"
            )

    # Step 1: For each ALTER file, identify the target MO list for that fragment.
    # A dict-based replacement is used so that swaps within one file are all
    # resolved against the *original* local active space — no sequential mutation.
    orb_map = {}
    for alter_file, local_active in zip(args.input, args.active_spaces):
        swaps = read_alter_swaps(alter_file)
        replacement = {}
        for target_orb, active_orb in swaps:
            if active_orb not in local_active:
                raise ValueError(
                    f"In {alter_file}, swap orbital {active_orb} is not in its declared local active space {local_active}"
                )
            replacement[active_orb] = target_orb
        # local_target is the post-swap orbital list for this fragment.
        local_target = [replacement.get(orb, orb) for orb in local_active]
        orb_map.update(zip(local_active, local_target))

    # Step 2: Assemble the combined target orbital list by substituting each
    # local active orbital with its post-swap counterpart in the global ordering.
    combined_target = [orb_map.get(orb, orb) for orb in total_active]

    write_alter_file(combined_target, total_active, args.output)
    print(f"Combined ALTER file written to {args.output}")


if __name__ == "__main__":
    main()
