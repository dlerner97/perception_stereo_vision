#<=============================== Imports ===================================>#
import cv2
from os import system

#<=============================== VideoBuffer Class Definition ===================================>#
class VideoBuffer():
    
    """
    Video Buffer Class
    
    Args: output video framerate
    
    This class handles video writing.
    """
    
    def __init__(self, framerate = 100):
        self.frames = []
        self.framerate = framerate

    # Write frames to feed and save video
    def save(self, video_name):
        shape = self.frames[0].shape
        size = (shape[1], shape[0])
        isColor=True
        
        try:
            if shape[2] < 3:
                isColor = False
        except IndexError:
            isColor = False
        
        if video_name[-4:] != '.avi':
            video_name = video_name + '.avi'
        
        videowriter = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), self.framerate, size, isColor)
        for f in self.frames:
            videowriter.write(f)
        videowriter.release()

#<=============================== VideoHandler Class Definition ===================================>#
class VideoHandler:
    
    """
    Video Handler Class
    
    Args: input video name, output, video name, scale factor, output video framerate
    
    Handles more complex video tasks such as generating video from images, resizing frames, and most importantly, running computer vision algorithms among all frames in a video feed.
    """
    
    def __init__(self, video_name, video_out_name, scale_percent=100, framerate=10) -> None:
        print("\nClick the image and select toggle 's' to pause/unpause the video between each frame")
        print("Click the image and select 'q' to cancel execution of the program.")
        self.video_feed = cv2.VideoCapture(video_name)
        self.scale_percent = scale_percent
        
        _, frame = self.video_feed.read()
        self.video_feed.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame = self.resize_frame(frame, self.scale_percent)
        self.shape = frame.shape[:2]
        self.video_out_name = video_out_name
        self.framerate = framerate
        
    # Static Method
    # Generates a video from a series of images
    @staticmethod
    def gen_video_frm_imgs(vid_name, img_rel_folder, name_char_len, last_num, img_type = '.png', framerate=10):
        buffer = VideoBuffer(framerate)
        
        if img_rel_folder[-1] != '/':
            img_rel_folder = img_rel_folder + '/'
        
        for i in range(last_num+1):
            str_i = str(i)
            img_name = img_rel_folder + '0'*(name_char_len-len(str_i)) + str_i + img_type
            img = cv2.imread(img_name)
            buffer.frames.append(img)
            
        buffer.save(vid_name)
        
    # Resizes each length image at a given percent. E.g. 200 will double each dimension and 50 will half it
    @staticmethod
    def resize_frame(frame, scale_percent=100):
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        return cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)    
    
    # Gets the nth frame without losing place in the video buffer
    def get_frame_n(self, frame_num):        
        curr_frame_num = self.video_feed.get(cv2.CAP_PROP_POS_FRAMES)
        self.video_feed.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        _, frame = self.video_feed.read()
        self.video_feed.set(cv2.CAP_PROP_POS_FRAMES, curr_frame_num)
        frame = self.resize_frame(frame, self.scale_percent)
        return frame
    
    # Returns img shape
    def get_img_shape(self):
        return self.shape
    
    # Returns number of frames in a video feed
    def get_num_frames(self):
        return self.video_feed.get(cv2.CAP_PROP_FRAME_COUNT)
    
    # Gets next frame in a video buffer
    def get_next_frame(self):
        ret, frame = self.video_feed.read()

        if not ret:
            return None, None
        
        frame = self.resize_frame(frame, self.scale_percent)
        return ret, frame
    
    # Runs computer vision algorithm "body()" for every frame in video, saves results in output video
    def run(self, body, orig_wait=1):
        wait = orig_wait
        buffer = VideoBuffer(self.framerate)
        try:
            while True:
                ret, frame = self.get_next_frame()

                if not ret:
                    break
            
                frame_processed = body(frame)
                buffer.frames.append(frame_processed)
                
                k = cv2.waitKey(wait) & 0xff
                if k == ord('q'):
                    raise KeyboardInterrupt
                elif k == ord('s'):
                    if wait == 0:
                        wait = orig_wait
                    else:
                        wait = 0
        except KeyboardInterrupt:
            self.video_out_name += "_INTERRUPTED"
        finally:
            buffer.save(self.video_out_name)
            print(f"Saved video in current directory as {self.video_out_name}.avi.")
            
    def __del__(self):
        self.video_feed.release()