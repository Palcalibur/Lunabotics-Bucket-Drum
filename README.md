# Lunabotics-Bucket-Drum
Purdue Lunabotics Excavation &amp; Deposition Sectioned Bucket Drum Design Repository

This repository contains a pre-compiled, custom Windows build of Project Chrono (PyChrono) specifically optimized for high-fidelity granular dynamics (CRM) and GPU offloading.
This build explicitly excludes the Chrono_vehicle modules to bypass CUDA compilation bugs, making it a lightweight and highly tailored environment for simulating robotic interactions (like a bucket drum) with lunar regolith.

⚠️ Hard Requirements:

To use this build, your system MUST meet the following requirements.
* Python 3.14 (64-bit): This build was compiled strictly against the Python 3.14 C-API. You must use a 64-bit Python 3.14 interpreter.

* NVIDIA GPU & CUDA 13.2 Drivers: You must have a dedicated NVIDIA GPU. While the CUDA runtime is statically linked, your NVIDIA display drivers must be updated to support CUDA 13.2.

* Windows 10/11 (x64)

🛠️ PyCharm Configuration Guide
Because this is a pre-compiled C++ library and not a standard pip package, you must manually point PyCharm to the included bin/Release folder.
Follow these exact steps to get IntelliSense (autocomplete) and run your simulations without "Module Not Found" errors:

Step 1: Set the Python 3.14 Interpreter

* Clone or download this repository to your local machine.

* Open the project folder in PyCharm.

* Go to File > Settings (or press Ctrl+Alt+S).

* Navigate to Project: [Your Project Name] > Python Interpreter.

* Click the Add Interpreter link (or the gear icon) and select Add Local Interpreter > System Interpreter.

* Browse to and select your Python 3.14 executable (e.g., C:\Users\<YourUser>\AppData\Local\Python\pythoncore-3.14-64\python.exe).

Step 2: Link the PyChrono Library

* Instead of using a system-wide PYTHONPATH, we will tell PyCharm exactly where to find the local binaries:
Still in the Python Interpreter settings menu, click the Gear Icon (or three dots) next to the interpreter dropdown and select Show All....

* Make sure your Python 3.14 interpreter is highlighted, then click the "Show paths for the selected interpreter" button at the top of the window (the icon looks like a small folder tree).

* Click the + (Add) button.

* Browse into your cloned repository folder and select the bin/Release folder.

* Click OK on all menus to apply the changes.

* If Pycharm Code Insight (intellisense) does not find pychrono (only a visual bug, code will run fine): Right click on bin/Release and click Mark Directory As -> Source Root
