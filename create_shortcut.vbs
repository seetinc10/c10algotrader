Set ws = CreateObject("WScript.Shell")
Set shortcut = ws.CreateShortcut(ws.SpecialFolders("Desktop") & "\AlgoTrader.lnk")
shortcut.TargetPath = "C:\Users\zbook 17\AppData\Local\Programs\Python\Python314\pythonw.exe"
shortcut.Arguments = "app.py"
shortcut.WorkingDirectory = "C:\c10algotrader2"
shortcut.Description = "C10 AlgoTrader - IBKR Paper Trading"
shortcut.WindowStyle = 1
shortcut.Save
WScript.Echo "Desktop shortcut created: AlgoTrader.lnk"
