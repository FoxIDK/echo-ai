import subprocess
import nltk

install = ["pip", "install", "-r", "requirements.txt"]
subprocess.run(install, shell=False)

nltk.download('punkt')
nltk.download('wordnet')

print("\n[!] REQUIREMENTS INSTALLED\n[!] Please run the: echo-l1.py inside of the directory 'src' to launch Echo.")