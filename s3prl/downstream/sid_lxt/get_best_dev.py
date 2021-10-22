import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--log", required=True)
args = parser.parse_args()

devs = []
tests = []
with open(args.log) as file:
    for line in file.readlines():
        line = line.strip()
        if "dev" in line:
            devs.append(float(line.split()[-1]))
        elif "test" in line:
            tests.append(float(line.split()[-1]))

assert len(devs) == 6
assert len(tests) == 6
print(tests[devs.index(max(devs))])
