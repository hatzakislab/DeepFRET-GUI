import os

os.system("fbs clean")
os.system("fbs freeze")
os.system("fbs installer")

print("Application compiled!")