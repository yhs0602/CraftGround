# Troubleshooting
### FileExistsError
```
FileExistsError: Socket file /tmp/minecraftrl_8001.sock already exists. Please choose another port.
```
Then
```bash
rm /tmp/minecraftrl_8001.sock 
```
### Zombie Minecraft process
```bash
jps -l # find the pid of something like DevLaunchInjector.Main
kill -9 <pid>
```