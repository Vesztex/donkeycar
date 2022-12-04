# Tools
Contains some files that are useful to have:
* __pi.bashrc__:  bashrc file with some useful aliases for the RPi
* __tub_data_plotter.ipynb__: jupyter notebook with simple pandas and graphing
    to display tub data
* To start `pigpiod` at boot use 
    ```bash
    sudo systemctl enable pigpiod
    sudo systemctl start pigpiod 
    ```
* __init_esc.py__: This initialises the ESC to stop the beeping. Best to copy:
    `/home/pi/env/bin/python3 /home/pi/init_esc.py` into `/etc/rc.local`. 
  [Here](https://www.dexterindustries.com/howto/run-a-program-on-your-raspberry-pi-at-startup/)
  is a page explaining different methods of running scripts at startup.
