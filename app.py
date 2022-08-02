import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import plotly.express as px
import plotly.graph_objects as go
# import pyttsx3
from threading import Thread
from flask import Flask, request, send_file


app = Flask(__name__)
# path = os.getcwd()


@app.route('/ballCount')
def ballCount():
    try:
        total_count=0
        url = request.args.get('url')
        video_path = str(url)
        # video_name = input("Enter video name e.g (myvideo.mp4): ")
#         video_path = path + '/input/' + str(url)


        print(video_path)
        # Threading Class
        # https://github.com/nateshmbhat/pyttsx3/issues/8
#         class Threader(Thread):
#             def __init__(self, *args, **kwargs):
#                 Thread.__init__(self, *args, **kwargs)
#                 self.daemon = True
#                 self.start()

#             def run(self):
#                 tts_engine = pyttsx3.init()
#                 tts_engine.setProperty('rate', 250)
#                 tts_engine.say(self._args)
#                 tts_engine.runAndWait()

        # curtosy of https://github.com/Enoooooormousb
        def ball_finder(frame, hsv_lower, hsv_upper):
            # blur the image to reduce noise
            blurred = cv2.GaussianBlur(frame, (11, 11), 0)
            #convert to hsv color space for color filtering
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            # construct a mask for the color specified
            mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
            # eliminate low color signal 
            mask = cv2.erode(mask, None, iterations=2)
            # expand the strong color signal
            mask = cv2.dilate(mask, None, iterations=2)
            return mask

        #count number of peaks in trace so far. Returns # of peaks
        def peak_calculator(height,cur_num_peaks):
            if len(height) > 9:
                #invert and filter input
                y = savgol_filter(height,9,2)
                #use peak finder
                peaks, _ = find_peaks(y)
                if len(peaks) < cur_num_peaks:
                    peaks = [cur_num_peaks]
            else:
                peaks = [0]
            return str(len(peaks))



        #HSV color limits of ball
        #https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-withcv
        colorLower = np.array([25,80,20])
        colorUpper = np.array([40,180,255])


        #save all the center coordinates
        x_centers = []
        y_centers = []

        # tally for speech to text
        count = '0'

        #Background Subtraction - pick which method to use
        #backSub = cv2.createBackgroundSubtractorMOG2()
        backSub = cv2.createBackgroundSubtractorKNN()
        background_subtract = True

        # Create a VideoCapture object and read from input file 
        cap = cv2.VideoCapture(video_path) #For pre-recorded
        #cap = cv2.VideoCapture(0) # for live video
        fps = cap.get(cv2.CAP_PROP_FPS)
        totalNoFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        dur = int(totalNoFrames / fps)


        #for saving video output
        capture_video = True
        # if capture_video:
        #     frame_width = int(cap.get(3))
        #     frame_height = int(cap.get(4))
        #     fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        #     out = cv2.VideoWriter('output/output.mp4',
        #                           cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))


        # Check if camera opened successfully 
        if (cap.isOpened()== False):  
            print("Error opening video file") 

        # Read until video is completed
        while(cap.isOpened()):
            # Capture frame-by-frame 
            ret, frame = cap.read() 
            if ret == True: 
                #resize the frame. This function keeps aspect ratio
                #frame = imutils.resize(frame, width=800)

                #if background subtraction
                if background_subtract:
                    #background subtract
                    fgMask = backSub.apply(frame)
                    #bitwise multiplication to grab moving part
                    new_frame = cv2.bitwise_and(frame,frame, mask = fgMask)


                # identify the ball
                mask = ball_finder(new_frame, colorLower, colorUpper)

                #canny edge detection
                edges = cv2.Canny(mask,100,200)
                # find contours of the ball
                contours, hierarchy = cv2.findContours(edges, 
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                #write frame number in top left
        #         cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        #         cv2.putText(frame, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
        #                cv2.FONT_HERSHEY_SIMPLEX, .5 , (0,0,0))

                new_frame = cv2.drawContours(new_frame, contours, -1, (0,255,0), 3)

                # get rid of excess contours and do analysis
                if len(contours) > 0:
                    # get the largest contour
                    c = max(contours, key=cv2.contourArea) 
                    # find the center of the ball
                    M = cv2.moments(c)
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    #save the center coordinates to a list
        #             x_centers.append(center[0])
                    y_centers.append(center[1])
                    total_count = peak_calculator(y_centers,int(count))

                    #draw center and contour onto frame
        #             cv2.circle(frame, center, 8, (255,255,0), -1)
        #             img = cv2.drawContours(frame, contours, -1, (0,0,255), 2)

                    #put number of peaks on frame
        #             cv2.putText(frame, peaks, (150,150), 
        #                         cv2.FONT_HERSHEY_SIMPLEX, 6, (0,0,255), 2)
        #             print(peaks)
                    #say outloud the count    
                    if count != total_count:
                        count = total_count
                        #if int(count) % 2 == 0:
                        #    my_thread = Threader(args = peaks)

                # Display the resulting frame 
        #         cv2.imshow('Frame', frame) 
                #cv2.imshow('Frame', new_frame)
                #cv2.imshow('Frame', edges)

                # write the flipped frame
        #         if capture_video:
        #             out.write(frame)


                # Press Q on keyboard to exit. 
                # waitkey is how long to show in ms frame
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                #elif cv2.waitKey(10) == ord(' '):
                #    y_centers = []
                #    count = '0'
                #    print('reset')

            # Break the loop 
            else:  
                break

        # When everything done, release  
        # the video capture object
        cap.release()
        # if capture_video:
        #     out.release()
        cv2.destroyAllWindows()

        if int(total_count)//4 > dur:
            return str(int(total_count)//4)

        elif int(total_count) <= dur:
            return str(int(dur + (dur * (50/100))))

        else:
            return str(int(total_count))
    except Exception as e:
        print(e)
        return str(e)

if __name__ == "__main__":
    app.run()
