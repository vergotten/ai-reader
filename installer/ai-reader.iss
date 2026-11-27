; Inno Setup 6 script. Build with installer\build.ps1 (it prepares build\stage\python first).
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#ifndef ModelsDir
  #define ModelsDir "..\models"
#endif
; voices shipped with the full build (build.ps1 copies them from the voice library); the app adds them to the
; user's library on first start
#ifndef VoicesDir
  #define VoicesDir "..\build\voices-pack"
#endif
; /DLite: Silero-only build with CPU PyTorch, small enough for one well-compressed .exe
#ifdef Lite
  #define Suffix "-Lite"
#else
  #define Suffix ""
#endif

[Setup]
AppId={{8F6B1C1E-5B7A-4E0B-9C55-A1D0C0FFEE01}
AppName=AI Reader
AppVersion={#AppVersion}
AppPublisher=AI Reader
; per-user install: no administrator rights, and the app may write models next to itself
PrivilegesRequired=lowest
DefaultDirName={localappdata}\Programs\AI Reader
DefaultGroupName=AI Reader
DisableProgramGroupPage=yes
; 64-bit Windows only (needs Inno Setup 6.3+; build.ps1 installs a newer one when missing)
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=AI-Reader-Setup-{#AppVersion}{#Suffix}
SetupIconFile=..\ai_reader\icon.ico
UninstallDisplayIcon={app}\ai_reader\icon.ico
WizardStyle=modern
#ifdef Lite
Compression=lzma2/max
SolidCompression=yes
#elif defined(Max)
; /DMax: the smallest full installer — strongest LZMA2 over everything as one solid stream. Takes hours;
; two compression threads keep the memory it needs at a few GB
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=2
DiskSpanning=yes
#else
; several GB of PyTorch and models: fast compression, split into 2 GB .bin files next to the .exe
Compression=lzma2/fast
SolidCompression=no
DiskSpanning=yes
#endif

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\build\stage\python\*"; DestDir: "{app}\python"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "..\ai_reader\*"; DestDir: "{app}\ai_reader"; Excludes: "__pycache__\*,*.pyc"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "{#ModelsDir}\*"; DestDir: "{app}\models"; Excludes: "xet\*,.locks\*,\torch\hub\checkpoints\*"; Flags: recursesubdirs createallsubdirs ignoreversion skipifsourcedoesntexist
#ifndef Lite
Source: "{#VoicesDir}\*"; DestDir: "{app}\voices"; Flags: recursesubdirs createallsubdirs ignoreversion skipifsourcedoesntexist
#endif
Source: "launch.pyw"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\AI Reader"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\launch.pyw"""; WorkingDir: "{app}"; IconFilename: "{app}\ai_reader\icon.ico"
Name: "{group}\{cm:UninstallProgram,AI Reader}"; Filename: "{uninstallexe}"
Name: "{userdesktop}\AI Reader"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\launch.pyw"""; WorkingDir: "{app}"; IconFilename: "{app}\ai_reader\icon.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\launch.pyw"""; WorkingDir: "{app}"; Description: "{cm:LaunchProgram,AI Reader}"; Flags: nowait postinstall skipifsilent

[InstallDelete]
; an update replaces the program and the models completely: old download-cache duplicates and models of
; removed engines (Piper, MMS, XTTS) go, the new set is installed below
Type: filesandordirs; Name: "{app}\ai_reader"
Type: filesandordirs; Name: "{app}\models\huggingface\hub"
Type: filesandordirs; Name: "{app}\models\torch\hub\checkpoints"
Type: filesandordirs; Name: "{app}\models\tts"
Type: filesandordirs; Name: "{app}\voices"

[UninstallDelete]
; files created after installation: models downloaded later, Python caches
Type: filesandordirs; Name: "{app}\models"
Type: filesandordirs; Name: "{app}\ai_reader"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\voices"
