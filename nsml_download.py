import argparse
import subprocess


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('user')
    parser.add_argument('dataset')
    parser.add_argument('sessions')
    parser.add_argument('target')
    parser.add_argument('-s', default=None)
    parser.add_argument('--no_block', action='store_true', default=False)
    args = parser.parse_args()
    if '-' in args.sessions:
        start, end = args.sessions.split('-')
        args.sessions = list(map(str, range(int(start), int(end)+1)))
    elif ',' in args.sessions:
        args.sessions = args.sessions.split(',')
    else:
        args.sessions = [args.sessions]
    return args


def run(cmd):
    proc = subprocess.Popen(cmd)
    return proc


def nsml_download(user, dataset, sessions, target, s, no_block):
    procs = []
    for session in sessions:
        source = f"{user}/{dataset}/{session}"
        if s is None:
            cmd = ["nsml", "download", source, target]
        else:
            cmd = ["nsml", "download", source, "-s", s, target]
        proc = subprocess.Popen(cmd)
        if no_block:
            procs.append(proc)
        else:
            out, err = proc.communicate()
            print(out.decode('utf-8'))

    for proc in procs:
        out, err = proc.communicate()
        print(out.decode('utf-8'))


def main():
    args = get_args()
    nsml_download(args.user, args.dataset, args.sessions, args.target, args.s, args.no_block)


if __name__ == "__main__":
    main()