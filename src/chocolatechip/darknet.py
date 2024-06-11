from cloudmesh.common.Shell import Shell
import yaspin
import time

def main():
    print('hello!')
    r = Shell.run("whoami")
    print(r)
    with yaspin.yaspin().white.bold.shark.on_blue as sp:
        sp.text = 'going to wait for the pipeline to finish'
        time.sleep(5)

if __name__ =="__main__":
    main()