import os
from project_paths import *
import cv2
from skimage.metrics import structural_similarity as ssim
from multiprocessing import Pool

folder_0 = train_original_frames_path / "0"
folder_1 = train_original_frames_path / "1"

folders = {str(0): folder_0, str(1): folder_1}

class DatasetAnalyzer:
    
    def compare_frames(frame1, frame2):
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        resized_frame1 = cv2.resize(frame1_gray, (640, 480))
        resized_frame2 = cv2.resize(frame2_gray, (640, 480))
        similarity = ssim(resized_frame1, resized_frame2)
        return similarity

    def compare_videos(video_pair):
        video1_frames, video2_frames = video_pair
        sim_tot = 0
        frame_count = 0

        for video1_frame in video1_frames:
            for video2_frame in video2_frames:
                similarity = DatasetAnalyzer.compare_frames(video1_frame, video2_frame)
                sim_tot += similarity
                frame_count += 1

        if frame_count > 0:
            similarita_media = sim_tot / frame_count
            return similarita_media
        else:
            return 0

    def compare_video_pair(args):
        video_pair, folder, type = args
        video1, video2 = video_pair
        # Resto del codice

        video1_path = str(folder) + "/" + video1
        video2_path = str(folder) + "/" + video2

        # Crea una lista di tutti i frame nel video 1
        video1_frames = []
        for frame in os.listdir(video1_path):
            frame_path = os.path.join(video1_path, frame)
            video1_frames.append(cv2.imread(frame_path))

        # Crea una lista di tutti i frame nel video 2
        video2_frames = []
        for frame in os.listdir(video2_path):
            frame_path = os.path.join(video2_path, frame)
            video2_frames.append(cv2.imread(frame_path))

        similarita_media = DatasetAnalyzer.compare_videos((video1_frames, video2_frames))
        
        # appende il risultato in un file csv
        with open('similarity_'+str(type)+'.csv', 'a') as f:
            f.write(video1 + "," + video2 + "," + str(similarita_media) + "\n")

if __name__ == '__main__':
    
    for num, folder in folders.items():
            
        video_pairs = []
        visited_videos = []
        for video1 in os.listdir(folder):
            for video2 in os.listdir(folder):
                if video1 != video2 and video2 not in visited_videos:
                    video_pairs.append((video1, video2))
            visited_videos.append(video1)

        # Creazione di un pool di processi con il numero di processi desiderati
        num_processes = 4
        pool = Pool(processes=num_processes)

        # Esecuzione parallela della funzione compare_video_pair
        pool.map(DatasetAnalyzer.compare_video_pair, [(video_pair, folder, num) for video_pair in video_pairs])

        # Chiusura del pool di processi
        pool.close()
        pool.join()
