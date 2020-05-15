"""!
Modified code from
https://towardsdatascience.com/image-scraping-with-python-a96feda8af2d
"""

import requests
from PIL import Image
import os
import io
import hashlib
from selenium import webdriver
import time
from webdriver_manager.chrome import ChromeDriverManager
from utilities.file_utilities import get_unique_filename
import face_recognition

first = True
known_face_encoding = None

def fetch_image_urls(query:str, max_links_to_fetch:int, wd:webdriver, sleep_between_interactions:int=1):
    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)    
    
    # build the google query
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # load the page
    wd.get(search_url.format(q=query))

    image_urls = set()
    image_count = 0
    results_start = 0
    while image_count < max_links_to_fetch:
        scroll_to_end(wd)

        # get all image thumbnail results
        thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)
        
        print(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")
        
        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            # extract image urls    
            actual_images = wd.find_elements_by_css_selector('img.n3VNCb')
            for actual_image in actual_images:
                if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                    image_urls.add(actual_image.get_attribute('src'))

            image_count = len(image_urls)

            if len(image_urls) >= max_links_to_fetch:
                print(f"Found: {len(image_urls)} image links, done!")
                break
        else:
            print("Found:", len(image_urls), "image links, looking for more ...")
            # time.sleep(30)
            # return
            load_more_button = wd.find_element_by_css_selector(".mye4qd")
            if load_more_button:
                wd.execute_script("document.querySelector('.mye4qd').click();")

        # move the result startpoint further down
        results_start = len(thumbnail_results)
        
    return image_urls

def persist_image(folder_path:str,url:str, first:bool):
    try:
        image_content = requests.get(url).content

    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")

    try:
        image_file = io.BytesIO(image_content)
        image = face_recognition.load_image_file(image_file)
    except Exception as e:
        print(e)
        return 1

    face_bounding_boxes = face_recognition.face_locations(image)

    if len(face_bounding_boxes) == 1: # If training image contains exactly one face
        # Find all the faces and face encodings in the current frame of video
        face_encoding = face_recognition.face_encodings(image, face_bounding_boxes)
        if first:
            # # Create a quick SVC classifer to remove any outliers pulled from Google image search
            # clf = svm.SVC(gamma='scale', probability=True)
            # name = folder_path.split("/").pop()
            # clf.fit(face_enc.reshape(1, -1), [name, "Unknown"])
            # save_model(name, clf)
            known_face_encoding = face_encoding
            first = False
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding[0], tolerance=0.7)
        if matches[0]:
            try:
                file_name = get_unique_filename(folder_path)
                # file_name = hashlib.sha1(image_content).hexdigest()[:10]
                file_path = os.path.join(folder_path, file_name + '.jpg')
                with open(file_path, 'wb') as f:
                    image = Image.open(image_file).convert('RGB') # Open the image
                    image.save(f, "JPEG", quality=90)
                print(f"SUCCESS - saved {url} - as {file_path}")
            except Exception as e:
                print(f"ERROR - Could not save {url} - {e}")
    elif len(face_bounding_boxes) == 0:
        print("WARNING: No face could be found in this picture. Skipping.")
    else:
        print("WARNING: More than one face is detected in this picture. Skipping.")

def search_and_download(search_term:str,driver_path:str,target_path='./images/training',number_images=5):
    # target_folder = os.path.join(target_path,'_'.join(search_term.lower().split(' ')))
    target_folder = target_path + "/" + search_term

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # with webdriver.Chrome(executable_path=driver_path) as wd:
    wd = driver_path
    res = fetch_image_urls(search_term, number_images, wd=wd, sleep_between_interactions=0.5)
    
    first = True
    try:
        for elem in res:
            persist_image(target_folder,elem, first)
    except Exception as err:
        print(err)

if __name__ == "__main__":
    driver = webdriver.Chrome(ChromeDriverManager().install())
    # names = ["Dakota Johnson", "Jamie Dornan"]
    # names = ["Ariana Grande", "Beyonce", "Chris Pratt", "Dwayne Johnson", "Justin Bieber", "Kim Kardashian", "Kylie Jenner", "Rihanna", "Selena Gomez", "Taylor Swift"]
    names = ["Ryan Gosling", "Emma Stone"]
    for name in names:
        if not os.path.isdir('./images/training'+"/"+name):
            search_and_download(name, driver, number_images=100)
    driver.close()