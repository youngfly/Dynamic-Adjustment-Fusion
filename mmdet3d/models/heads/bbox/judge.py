import yaml
with open("transfusion_inter.py") as f:
    with open("demo.py") as k:
        while(1):
            x=f.readline()
            if(x):
                if(x==k.readline()):
                    pass
                else:
                    print("not equal")
                    print(x)
            else:
                print("end!")
                break