# 通过 RLG 反向 SSH 访问 Windows

本文记录从服务器访问 Windows 主机的配置。连接方向是：Windows 主动连接服务器，服务器再通过反向端口转发访问 Windows 的 SSH 服务。

```text
Windows  --ssh -R-->  RLG 服务器:63173
                         |
                         └── 127.0.0.1:2222 -> Windows:22
```

以下命令假设：

- Windows SSH 配置中已有 `Host RLG`；
- Windows 用户名为 `oyx`；
- 服务器上的公钥为 `~/.ssh/id_ed25519.pub`，私钥为 `~/.ssh/id_ed25519`；
- 反向转发端口使用服务器本机的 `2222`。

## 1. 在 Windows 启动反向隧道

在 Windows PowerShell 中运行，并保持该窗口不退出：

```powershell
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -R 127.0.0.1:2222:127.0.0.1:22 RLG
```

服务器上检查端口：

```bash
ss -ltn | rg ':2222\b'
```

应看到 `127.0.0.1:2222` 处于监听状态。该绑定只允许服务器本机访问，不要改成 `0.0.0.0`，除非确实需要对外暴露。

## 2. 配置服务器公钥登录 Windows

Windows OpenSSH Server 必须已经运行。若 `oyx` 属于 Windows 的 Administrators 组，请在“以管理员身份运行”的 PowerShell 中执行：

```powershell
# 该命令会要求输入服务器账户 wyy 的密码；密码输入时不会显示字符
$pub = (ssh RLG "cat ~/.ssh/id_ed25519.pub").Trim()
$auth = "$env:ProgramData\ssh\administrators_authorized_keys"

New-Item -ItemType Directory -Force (Split-Path $auth) | Out-Null
$lines = if (Test-Path -LiteralPath $auth) { @(Get-Content -LiteralPath $auth) } else { @() }
if ($lines -notcontains $pub) { $lines += $pub }
$utf8 = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllLines($auth, $lines, $utf8)

icacls $auth /inheritance:r
icacls $auth /grant '*S-1-5-32-544:F' '*S-1-5-18:F'
```

如果 `oyx` 不是管理员，改用普通用户密钥文件：

```powershell
$pub = (ssh RLG "cat ~/.ssh/id_ed25519.pub").Trim()
$auth = "$env:USERPROFILE\.ssh\authorized_keys"

New-Item -ItemType Directory -Force (Split-Path $auth) | Out-Null
$lines = if (Test-Path -LiteralPath $auth) { @(Get-Content -LiteralPath $auth) } else { @() }
if ($lines -notcontains $pub) { $lines += $pub }
$utf8 = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllLines($auth, $lines, $utf8)
```

如果第一行等待密码，输入服务器账户 `wyy` 的密码并按 Enter；不要把密码写入文档或发送给其他人。

## 3. 在服务器验证连接

Windows OpenSSH 默认使用 `cmd.exe`，远程命令使用 `&&` 连接，不要使用 Linux 风格的 `;`、`ls`：

```bash
ssh -p 2222 -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes oyx@127.0.0.1 "whoami && cd && dir"
```

预期结果类似：

```text
oyx\oyx
C:\Users\oyx
```

## 4. 访问 Windows 文件

查看文件或目录：

```bash
ssh -p 2222 -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes oyx@127.0.0.1 'dir /B "C:\Users\oyx\Documents"'
ssh -p 2222 -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes oyx@127.0.0.1 'type "C:\Users\oyx\Documents\example.txt"'
```

使用 SFTP 交互传输：

```bash
sftp -P 2222 -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes oyx@127.0.0.1
```

进入 SFTP 后，Windows 路径使用 `/C:/` 开头：

```text
pwd
cd /C:/Users/oyx/Documents
ls
get example.txt
put local.txt /C:/Users/oyx/Downloads/local.txt
bye
```

## 5. 常见问题

- `127.0.0.1:2222` 不监听：Windows 上的反向隧道窗口已退出，重新运行第 1 节命令。
- `Permission denied`：确认 Windows 用户名、密钥文件位置，以及管理员账户是否使用了 `C:\ProgramData\ssh\administrators_authorized_keys`。
- 中文输出乱码：这是 Windows 命令行代码页问题，不代表连接失败；可以在远程命令前加 `chcp 65001 >NUL &&`。
- 结束连接：在 Windows 反向隧道窗口按 `Ctrl+C`。
