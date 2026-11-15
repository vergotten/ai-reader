; Inno Setup 6 script. Build with installer\build.ps1 (it prepares build\stage\python first).
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#ifndef ModelsDir
  #define ModelsDir "..\models"
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
; "x64" rather than "x64compatible": also understood by Inno Setup older than 6.3
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
OutputDir=..\dist
OutputBaseFilename=AI-Reader-Setup-{#AppVersion}{#Suffix}
SetupIconFile=..\ai_reader\icon.ico
UninstallDisplayIcon={app}\ai_reader\icon.ico
WizardStyle=modern
#ifdef Lite
Compression=lzma2/max
SolidCompression=yes
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
Source: "launch.pyw"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\AI Reader"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\launch.pyw"""; WorkingDir: "{app}"; IconFilename: "{app}\ai_reader\icon.ico"
Name: "{group}\{cm:UninstallProgram,AI Reader}"; Filename: "{uninstallexe}"
Name: "{userdesktop}\AI Reader"; Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\launch.pyw"""; WorkingDir: "{app}"; IconFilename: "{app}\ai_reader\icon.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\python\pythonw.exe"; Parameters: """{app}\launch.pyw"""; WorkingDir: "{app}"; Description: "{cm:LaunchProgram,AI Reader}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; files created after installation: models downloaded later, Python caches
Type: filesandordirs; Name: "{app}\models"
Type: filesandordirs; Name: "{app}\ai_reader"
Type: filesandordirs; Name: "{app}\python"
