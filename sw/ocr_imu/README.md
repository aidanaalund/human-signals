### Comments from Michael

Python code and also datasets are inside the python folder.
See readme inside the python folder for more details about everything inside there.

Arduino code is inside the arduino folder.


Notes during development:
- I could only get the IMU to stream IMU data smoothly and reliably at 52 Hz. Ideally I'd want 104 Hz, but I was getting chunks of data updating at seemly 1 Hz intervals.
- I chose to use baud rate 230400 but I think 115200 will work just fine. I beleive the bottleneck is not the serial communication, but rather the I2C sensor reading from within the MG24 itself. 
- I tried reading directly from the FIFO but then I was getting ramp artifacts in the data, possibly due to offset bits or misalignment in how I am reading the data? Not sure. I just went ahead and worked with 52 Hz sampling rate though if we could get higher, that would be better. Phone IMU apps typically have 100 Hz sampling rate.

- I did not do any wireless development. I only focused on IMU and tried to see what is possible.
- Some of the code is not very cleaned up or annotated, but I can clean up if needed.
- Some of the code is inspired by AI, but modified for our use.



