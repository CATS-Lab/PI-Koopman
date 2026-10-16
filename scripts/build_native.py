"""Compile the unchanged C99 RK4 implementation; no prebuilt binary shipped."""
import os, pathlib, shlex, subprocess, sys
root=pathlib.Path(__file__).resolve().parents[1]
def main():
    if sys.platform not in ('darwin','linux'):
        raise SystemExit('Supported native targets: macOS and Linux; on Windows use WSL.')
    out=root/'build';out.mkdir(exist_ok=True)
    name='revision_native.dylib' if sys.platform=='darwin' else 'revision_native.so'
    cmd=shlex.split(os.environ.get('CC','cc'))+['-O3','-std=c99','-fPIC',
        '-dynamiclib' if sys.platform=='darwin' else '-shared',
        str(root/'src/models/native.c'),'-o',str(out/name),'-lm']
    subprocess.run(cmd,check=True)
    print(out/name)


if __name__ == "__main__":
    main()
