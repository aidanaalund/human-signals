# Preliminary IMU Exploration

-----
### Structure

Note: These were copy-pasted from my messy local directory so some of the file paths may not run as is, but can easily be fixed/changed to reflect the new naming/structure.

```datasets [dir]```
- This directory contains the prompted_dataset_1, which is a custom created dataset by yours truly (Michael). The dataset supports several 'sessions' or users, but I just put everything in 1 session, but later can be used to distinguish users. Inside the session, there is a folder for each character and digit, case insensitive. Inside each folder however, there are csv files of each individual sample of the corresponding character being written, which is case sensitive as denoted by the first character of the csv file name. (Yes this is a bug and ideally each dir should be their own class but I am too lazy to fix/change it).
- 62 classes: A-Z, a-z, 0-9.

- Important note about the prompted_dataset_1 data. I am here to confess that I messed up a few characters here and there. Especially for lowercase l, at times I wrote capital I. At times, I just accidentally wrote the wrong character than what I was prompted. This was maybe no more than 5 samples I wrote the wrong character. But a significant number of 'l' was written as 'I'. Additionally, my '1' and 'l' class is virtually the same handwriting (vertical line). In the future, I'd probably make the '1' serif. Whatever. This was just for proof of concept and initial testing. May redo in the future if desired.

```saved_models [dir]```
- This directory contains folders representing different saved models from the ```char_recognition``` notebook containing metadata of the parameters used as well as labels and the .keras file for the actual model itself.

```char_recognition```
- This is the main python notebook (ipynb) for ML model exploration. This loads data from the '''datasets''' dir and then does data processing, minimal feature extraction (calculate magnitude) and then explores different 1D CNN models built with keras TF. Good models are then saved to the ```saved_models``` to be loaded by the ```realtime-inference``` script.


```prompted_record```
- Randomly generates (with set seed) a list of 3720 characters (62 classes x 60 samples each) to prompt the user to write the character one by one with the IMU pen. When space bar is pressed, recording starts, and when space bar is released, recording stops and the data is labeled and saved according to the specified dataset parent directory and the prompted character on screen. For example, if the script prompts the user to write 'X', then the saved data on space bar release will be labeled 'X'. Data is saved as a csv file containing accel and gyro data.
- Note: separate directories within the parent dataset dir will be created to organize the individual csv files. However, the directories will ignore case, but the actual csv files have the correct case. For example, both 'H' and 'h' will be stored in the 'H' directory or the 'h' directory, whichever is created first.
- Note: this script also prompts the user to write at different speeds and also logs a speed label, as I thought this would be useful, but at the end, I just ignored this during my data collection and this label is now useless.
- IMU stream will also be displayed for user convenience and debugging.

```realtime-inference```
- This script reads in IMU data via serial USB and attempts to window the data dynamically based on thresholding and perform inference using a model loaded from ```saved_models```. It will print the predicted character and its confidence, along with the window size.

```record_data```
- This script is the manual version of prompted_record. Instead of pressing and releasing space, whatever key you press, it will record and label accordingly. For example if you press and release the 'a' key, it will label the recorded data as 'a'.

```stream_imu```
- This script reads in IMU data via serial USB from the XIAO MG24 SENSEand displays the data in an interactive plot that updates in real time using PyQt. The sampling rate from the sensor is 52 Hz but the display will update at 50 Hz. (No particular reason why I chose to not have them the same, it was an artifact from just testing and stuff and was lazy and still too lazy to change it)







