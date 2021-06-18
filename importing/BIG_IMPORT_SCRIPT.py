import selenium
from selenium import webdriver
import time
import threading
import urllib.request
import os
from selenium.webdriver.common.keys import Keys
import selenium.webdriver.chrome.options as chropt
from selenium.common.exceptions import StaleElementReferenceException


def start_chrome():
    options = chropt.Options()
    options.add_argument("--start-maximized")
    return webdriver.Chrome(
        r"D:/Coding/TensorflowLearning/Licenta/DatasetImporting/chromedriver_win32/chromedriver.exe",
        options=options)


def unsplash_import(chrome, label, path):
    global src_set
    global img_nr
    global max_per_label
    chrome.get("https://unsplash.com/")
    search = chrome.find_element_by_name("searchKeyword")
    search.send_keys(label, Keys.ENTER)
    time.sleep(3)
    value = 0
    count = 0
    skip_count = 0
    while skip_count <= 20 and img_nr < 100000 and count < max_per_label:
        found = 0
        try:
            list_of_img = chrome.find_elements_by_tag_name(
                "img")
            if list_of_img:
                for image in list_of_img:
                    if image:
                        src = image.get_attribute("src")
                        try:
                            if src is not None and src not in src_set:
                                src = str(src)
                                src_set.add(src)
                                try:
                                    src = str(src)
                                    try:
                                        urllib.request.urlretrieve(src, os.path.join(path,
                                                                                     str(img_nr) + '.jpg'))
                                    except urllib.error.URLError:
                                        pass
                                    src_set.add(src)
                                    found += 1
                                    skip_count = 0
                                    img_nr += 1
                                    count += 1
                                except Exception:
                                    pass
                            else:
                                raise TypeError
                        except TypeError:
                            pass
        except StaleElementReferenceException:
            continue
        if found == 0:
            skip_count += 1
            value += 250
            chrome.execute_script("scrollBy(" + str(value) + ", +250);")


def google_import(chrome, label, path):
    global src_set
    global img_nr
    global max_per_label
    chrome.get('https://www.google.ro/imghp?hl=en&tab=wi&ogbl')
    search = chrome.find_element_by_name("q")
    search.send_keys(label, Keys.ENTER)
    time.sleep(3)
    value = 0
    count = 0
    skip_count = 0
    while skip_count <= 20 and img_nr < 100000 and count < max_per_label:
        elem1 = chrome.find_element_by_id('islmp')
        batch = elem1.find_elements_by_tag_name('img')
        found = 0
        for i in batch:
            src = i.get_attribute('src')
            try:
                if src is not None and src not in src_set:
                    src = str(src)
                    try:
                        urllib.request.urlretrieve(src, os.path.join(path,
                                                                     str(img_nr) + '.jpg'))
                    except urllib.error.URLError:
                        pass
                    src_set.add(src)
                    found += 1
                    skip_count = 0
                    img_nr += 1
                    count += 1
                else:
                    raise TypeError
            except TypeError:
                pass
        if found == 0:
            try:
                skip_count += 1
                more = chrome.find_element_by_class_name("mye4qd")
                more.click()
                time.sleep(3)
            except Exception:
                pass
        value += 250
        chrome.execute_script("scrollBy(" + str(value) + ", +250);")


def yahoo_import(chrome, label, path):
    global src_set
    global img_nr
    global max_per_label
    chrome.get('https://images.search.yahoo.com/')
    try:
        agree = chrome.find_element_by_name("agree")
        agree.click()
    except selenium.common.exceptions.NoSuchElementException:
        pass
    search = chrome.find_element_by_name("p")
    search.send_keys(label, Keys.ENTER)
    time.sleep(3)
    count = 0
    value = 0
    skip_count = 0
    while skip_count <= 20 and img_nr < 100000 and count < max_per_label:
        found = 0
        list_items = chrome.find_elements_by_tag_name('li')
        for element in list_items:
            try:
                img = element.find_elements_by_tag_name("img")
                if img:
                    src = img[0].get_attribute('src')
                    try:
                        if src is not None and src not in src_set:
                            src = str(src)
                            try:
                                urllib.request.urlretrieve(src, os.path.join(path,
                                                                             str(img_nr) + '.jpg'))
                            except urllib.error.URLError:
                                pass
                            src_set.add(src)
                            found += 1
                            skip_count = 0
                            img_nr += 1
                            count += 1
                        else:
                            raise TypeError
                    except TypeError:
                        pass
            except StaleElementReferenceException:
                continue
        if found == 0:
            try:
                skip_count += 1
                more = chrome.find_element_by_name("more-res")
                more.click()
                time.sleep(3)
            except Exception:
                pass
        value += 250
        chrome.execute_script("scrollBy(" + str(value) + ", +250);")


def getty_import(chrome, label, path):
    global src_set
    global img_nr
    global max_per_label
    chrome.get('https://www.gettyimages.com/')
    search = chrome.find_element_by_name("phrase")
    search.send_keys(label, Keys.ENTER)
    time.sleep(3)
    value = 0
    count = 0
    skip_count = 0
    while skip_count <= 20 and img_nr < 100000 and count < max_per_label:
        found = 0
        list_items = chrome.find_elements_by_class_name(
            "gallery-mosaic-asset__figure")
        if list_items:
            for element in list_items:
                try:
                    image = element.find_elements_by_tag_name("img")
                    src = image[0].get_attribute("src")
                    try:
                        if src is not None and src not in src_set:
                            src = str(src)
                            try:
                                urllib.request.urlretrieve(src, os.path.join(path,
                                                                             str(img_nr) + '.jpg'))
                            except urllib.error.URLError:
                                pass
                            src_set.add(src)
                            found += 1
                            skip_count = 0
                            img_nr += 1
                            count += 1
                        else:
                            raise TypeError
                    except TypeError:
                        pass
                except StaleElementReferenceException:
                    continue
        if found == 0:
            try:
                skip_count += 1
                more = chrome.find_elements_by_class_name("search-pagination")
                more = more[0].find_elements_by_tag_name("a")
                href = more[0].get_attribute("href")
                chrome.get(href)
                time.sleep(3)
            except Exception:
                pass
        value += 250
        chrome.execute_script("scrollBy(" + str(value) + ", +250);")


def shutter_import(chrome, label, path):
    global src_set
    global img_nr
    global max_per_label
    chrome.get('https://www.shutterstock.com/')
    search = chrome.find_element_by_name("searchterm")
    search.send_keys(label, Keys.ENTER)
    time.sleep(3)
    value = 0
    page = 1
    count = 0
    skip_count = 0
    initial_url = None
    while skip_count <= 20 and img_nr < 100000 and count < max_per_label:
        found = 0
        list_items = chrome.find_elements_by_class_name(
            "z_g_63ded")
        if list_items:
            for element in list_items:
                try:
                    image = element.find_elements_by_tag_name("img")
                    src = image[0].get_attribute("src")
                    try:
                        if src is not None and src not in src_set:
                            src = str(src)
                            try:
                                urllib.request.urlretrieve(src, os.path.join(path,
                                                                             str(img_nr) + '.jpg'))
                            except urllib.error.URLError:
                                pass
                            src_set.add(src)
                            found += 1
                            skip_count = 0
                            img_nr += 1
                            count += 1
                        else:
                            raise TypeError
                    except TypeError:
                        pass
                except StaleElementReferenceException:
                    continue
        if found == 0:
            try:
                skip_count += 1
                url = "?page="
                if page == 1:
                    initial_url = chrome.current_url
                    page = 2
                else:
                    page += 1
                url = initial_url + url + str(page)
                chrome.get(url)
                time.sleep(3)
            except Exception:
                pass
        value += 250
        chrome.execute_script("scrollBy(" + str(value) + ", +250);")


def bing_import(chrome, label, path):
    global src_set
    global img_nr
    global max_per_label
    chrome.get("https://www.bing.com/?scope=images&nr=1&FORM=NOFORM")
    search = chrome.find_element_by_name("q")
    search.send_keys(label, Keys.ENTER)
    imglink = chrome.find_element_by_link_text("IMAGES")
    chrome.get(imglink.get_attribute("href"))

    value = 0
    skip_count = 0
    count = 0
    while skip_count <= 20 and img_nr < 100000 and count < max_per_label:
        found = 0
        list_of_lists = chrome.find_elements_by_tag_name(
            "ul")
        if list_of_lists:
            for list1 in list_of_lists:
                try:
                    li_list = list1.find_elements_by_tag_name("li")
                    if li_list:
                        try:
                            for element in li_list:
                                image = element.find_elements_by_tag_name("img")
                                if image:
                                    src = image[0].get_attribute("src")
                                    try:
                                        if src is not None and src not in src_set:
                                            src = str(src)

                                            try:
                                                urllib.request.urlretrieve(src, os.path.join(path,
                                                                                             str(img_nr) + '.jpg'))
                                            except urllib.error.URLError:
                                                pass
                                            src_set.add(src)
                                            found += 1
                                            skip_count = 0
                                            img_nr += 1
                                            count += 1
                                        else:
                                            raise TypeError
                                    except TypeError:
                                        pass
                        except StaleElementReferenceException:
                            continue
                except StaleElementReferenceException:
                    continue
        if found == 0:
            skip_count += 1
            value += 250
            chrome.execute_script("scrollBy(" + str(value) + ", +250);")


def start_imports(chrome, label_list, function, path):
    for label in label_list:
        function(chrome, label, path)


src_set = set()
img_nr = 0
max_per_label = 2000

if __name__ == '__main__':
    # global img_nr
    labels = ["zucchini","courgette","baby marrow",
              "zucchini fruit", "courgette fruit", "baby marrow fruit",
              "zucchini plant", "courgette plant", "baby marrow plant",
              "zucchini fruit plant", "courgette fruit plant", "baby marrow fruit plant",
              ]
    with open("../Dataset/zucchini.txt", "w") as o:
        o.write(str(labels))
    os.mkdir("../CleanedDataset/zucchini")
    chrome1 = start_chrome()
    chrome2 = start_chrome()
    chrome3 = start_chrome()
    chrome4 = start_chrome()
    chrome5 = start_chrome()
    chrome6 = start_chrome()
    t1 = threading.Thread(target=start_imports,
                          args=[chrome1, labels, unsplash_import, '../Dataset/data/unsplash/'])
    t2 = threading.Thread(target=start_imports,
                          args=[chrome2, labels, google_import, '../Dataset/data/google/'])
    t3 = threading.Thread(target=start_imports, args=[chrome3, labels, yahoo_import, '../Dataset/data/yahoo/'])
    t4 = threading.Thread(target=start_imports, args=[chrome4, labels, getty_import, '../Dataset/data/getty/'])
    t5 = threading.Thread(target=start_imports,
                          args=[chrome5, labels, shutter_import, '../Dataset/data/shutter/'])
    t6 = threading.Thread(target=start_imports, args=[chrome6, labels, bing_import, '../Dataset/data/bing/'])
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t6.start()
