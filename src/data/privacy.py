"""Remove absolute navigation/time fields from a generated CATS-CF release."""
from pathlib import Path
import csv


def is_private(name):
    return (name in {'t_gps_s', 't_abs_s', 'source_trial_id'}
            or name.startswith('window_start_gps')
            or name.startswith('window_end_gps')
            or name.startswith('enu_origin_')
            or ('latitude' in name and 'stdev' not in name)
            or ('longitude' in name and 'stdev' not in name))


def sanitize_release(root):
    root = Path(root)
    for path in sorted(root.rglob('*.csv')):
        temporary = path.with_suffix('.csv.tmp')
        with path.open(newline='') as source, temporary.open('w', newline='') as target:
            reader = csv.DictReader(source)
            fields = [key for key in reader.fieldnames if not is_private(key)]
            writer = csv.DictWriter(target, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
            writer.writeheader()
            for row in reader:
                if 'column' in row and is_private(row['column']):
                    continue
                # Relative time remains unchanged; dictionary no longer names the removed field.
                if row.get('column') == 't_rel_s':
                    row['source'] = 'synchronized sample time'
                    row['transformation'] = 'sample time minus selected window start'
                writer.writerow(row)
        temporary.replace(path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    sanitize_release(parser.parse_args().directory)
