import re

data = open("metrics.txt").read()

inf, api, server, fps = [], [], [], []

for line in data.splitlines():
    m = re.search(r"Inf: ([\d.]+).*Pre: ([\d.]+).*Post: ([\d.]+).*Enc: ([\d.]+).*FPS: ([\d.]+)", line)
    if m:
        i, pre, post, enc, f = map(float, m.groups())

        inf.append(i)
        api.append(pre + i + post)
        server.append(pre + i + post + enc)
        fps.append(f)

print("Inference:", sum(inf)/len(inf))
print("API:", sum(api)/len(api))
print("Server:", sum(server)/len(server))
print("FPS:", sum(fps)/len(fps))