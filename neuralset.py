import subprocess

install = ["pip", "install", "-r", "requirements.txt"]
subprocess.run(install)

print("[!] REQUIREMENTS INSTALLED\n[!] Please run the: echo-l1.py inside of the directory 'src' to launch Echo.")