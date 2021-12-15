# PFNN_TVM

Efficient PFNN implementations enabled by TVM. 

This description sketches 
* how to use tuned scheduler to build PFNN for good performance.
* how to tune PFNN on diffierent mobile phones.

---
## How to build
**Install required build dependencies:**
* git 
* python3
* cmake: version >= 3.2
* TVM (The existing scheduler are collected by git commit 970aeff)

TVM is used to generate scheduler with good performance. Please follow the tutorial https://tvm.apache.org/docs/install/from_source.html to install TVM .   

**Git clone pfnn_tvm repo**
```bash
git clone https://github.com/turbo0628/pfnn_tvm
cd PFNN_TVM
```
### Build for Linux
**Build source:**
```bash
bash ./scripts/linux/build.sh <mode> <interp> <use_extern> <tuned_logs_dir>
# Example
# bash ./scripts/linux/build.sh ansor A no_extern ./tuned_logs/8255c_tuned_logs
```
Script options:
  - mode :   
    - manual : Use naive manual tuning methods  
    - autoTVM : Use the methods tuned by autoTVM   
    - ansor : Use the methods tuned by ansor 
  - interp : 
    - A : Use the methods of interpolation A
    - B : Use the methods of interpolation B
  - use_extern : 
    - no_extern : Use the default TVM scheduler as inner kernel
    - extern : Use developed embedded inner kernel
  - tuned_logs_dir : 
    - Tuned logs directory path  
  
**Run executable:**
```bash
cd build
./pfnn_benchmark -m ../models -b <1/5/8> -r 200
```
Running options:
- -m, --model : Model directory path (default: ./models)
- -b, --batch : Batch size (default: 5)
- -r, --repeats : Repeat tests for benchmark (default: 200)


### Build for Android
This program requires hardwares support `armv8` architecture.  

Android ADB is used to upload the binary executable file to the mobile device and remotely debug the program on the server. Please install Android ADB from https://developer.android.com/studio/command-line/adb.  

Android NDK is used to cross-compile the program on the server.Please install Android NDK from https://developer.android.com/ndk/guides.  


**Build source:**
```bash
export NDK_ROOT=${your_ndk_root}
bash ./scripts/android/build.sh  <mode> <interp> <use_extern> <tuned_logs_dir>
# Example
# bash ./scripts/android/build.sh ansor A no_extern ./tuned_logs/855_tuned_logs
```


**Upload files:**
```bash
bash ./scripts/android/adb_push.sh ${Path_in_android_device}
```

**Run executable on android device:**
```bash
# Connect to android device
adb shell
# The following commands are on android device
cd ${Path_in_android_device}
./pfnn_benchmark -m ./models -b <1/5/8> -r 200
```
---
## How to tune

If you want to tune `ansor` mode, you need to reference the QA (https://discuss.tvm.apache.org/t/number-of-threads-used-during-auto-tunning/570) modify the TVM source code so that ansor run can run on a single core.

### Tune for Linux

**Begin tune process:**
```bash
bash ./scripts/android/tune.sh  <mode(autoTVM/ansor)> <interp(A/B)> <use_extern(no_extern/extern)>
# Example
# bash ./scripts/linux/tune.sh ansor A no_extern 
```

### Tune for Android
**Install `tvmrpc-release.apk` on android device, follow the tutorial:**
- https://github.com/apache/tvm/tree/main/jvm  
- https://github.com/apache/tvm/tree/main/apps/android_rpc  


Open the app, set the Address and Port fields to the address and port of the RPC tracker respectively. The key should be set to 'android" if you wish to avoid modifying the default test script.
* Address=${your_tracker_host_address}  
* Port=${your_tracker_port}
* Key=android  

**Checkout RPC:**
```bash
export TVM_TRACKER_HOST=0.0.0.0
export TVM_TRACKER_PORT=${your_tracker_port}
python -m tvm.exec.rpc_tracker --host=${TVM_TRACKER_HOST} --port=${TVM_TRACKER_PORT} &
python -m tvm.exec.query_rpc_tracker --host=${TVM_TRACKER_HOST} --port=${TVM_TRACKER_PORT}
# Shows like following
# 
# Queue Status
# -------------------------------
# key       total  free  pending
# -------------------------------
# android   1      1     0      
# -------------------------------
```

**Begin tune process:**
```bash
export NDK_ROOT=${your_ndk_root}
bash ./scripts/android/tune.sh  <mode(autoTVM/ansor)> <interp(A/B)> <use_extern(no_extern/extern)>
# Example
# bash ./scripts/android/tune.sh ansor A no_extern 
```

