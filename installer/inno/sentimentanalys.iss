; Inno Setup script for Automatisk-sentimentanalys
; Build: ISCC.exe sentimentanalys.iss /DMyAppVersion=0.4.1 /DInstallProfile=full

#ifndef MyAppVersion
  #define MyAppVersion "0.4.1"
#endif
#ifndef InstallProfile
  #define InstallProfile "full"
#endif

#define MyAppName "Sentimentanalys"
#define MyAppPublisher "ideal-invest"
#define MyAppURL "https://github.com/ideal-invest/Automatisk-sentimentanalys"
#define MyAppExeName "Sentimentanalys.bat"

[Setup]
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=no
OutputDir=..\..\dist
OutputBaseFilename=Sentimentanalys-Setup-{#MyAppVersion}-{#InstallProfile}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "swedish"; MessagesFile: "compiler:Languages\Swedish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "removeuserdata"; Description: "Ta bort användardata i %%AppData%%\Sentimentanalys (config, loggar, state)"; GroupDescription: "Avinstallation:"; Flags: unchecked

[Files]
Source: "..\..\build\portable-staging\Sentimentanalys\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName} Launcher"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{group}\Doctor"; Filename: "{app}\.venv\Scripts\python.exe"; Parameters: "-m launcher.cli doctor"; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; WorkingDir: "{app}"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: postinstall nowait skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{userappdata}\Sentimentanalys"; Tasks: removeuserdata

[Code]
function InitializeSetup(): Boolean;
begin
  if not DirExists(ExpandConstant('{src}\..\..\build\portable-staging\Sentimentanalys')) then
  begin
    MsgBox('Portable staging not found. Run installer\build-portable.ps1 first.', mbError, MB_OK);
    Result := False;
  end
  else
    Result := True;
end;