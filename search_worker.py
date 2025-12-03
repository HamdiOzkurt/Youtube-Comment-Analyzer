"""
YouTube Video URL Arama Worker
Selenium ile YouTube'da arama yapar ve video URL'lerini toplar
"""

import time
import urllib.parse
import sys

from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, NoSuchElementException, WebDriverException,
    SessionNotCreatedException, InvalidArgumentException
)

try:
    from webdriver_manager.chrome import ChromeDriverManager
    USE_WEBDRIVER_MANAGER = True
except ImportError:
    print("⚠️  webdriver_manager bulunamadı. 'pip install webdriver-manager' çalıştırın.")
    USE_WEBDRIVER_MANAGER = False

from PyQt6.QtCore import QObject, pyqtSignal


class SearchWorker(QObject):
    """Selenium ile YouTube video arama worker'ı"""
    
    # Sinyaller
    search_finished = pyqtSignal(list)  # Bulunan URL listesi
    search_error = pyqtSignal(str)      # Hata mesajı

    def __init__(self, query, limit, lang=None, parent=None):
        super().__init__(parent)
        self.query = query
        self.limit = limit
        self.lang = lang
        self._is_running = True
        self.driver = None
        self.found_urls = []

    def stop(self):
        """Worker'ı durdur"""
        print("⏹️  Arama durdurma isteği alındı...")
        self._is_running = False

    def run(self):
        """Selenium ile YouTube araması yapar"""
        self._is_running = True
        self.found_urls = []

        print(f"🔍 Arama başlatılıyor: '{self.query}' için {self.limit} video aranıyor...")

        # Chrome ayarları
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--log-level=3')
        options.add_argument('--disable-gpu')
        
        # Dil ayarı varsa user-agent veya header ile desteklenebilir ama
        # URL parametresi (?hl=en) en garantisidir.
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)

        try:
            # ChromeDriver başlat
            print("🚗 ChromeDriver başlatılıyor...")
            if USE_WEBDRIVER_MANAGER:
                service = Service(ChromeDriverManager().install())
                self.driver = webdriver.Chrome(service=service, options=options)
            else:
                self.driver = webdriver.Chrome(options=options)

            if not self._is_running:
                raise Exception("Başlamadan durduruldu")

            # Arama sayfasına git
            encoded_query = urllib.parse.quote_plus(self.query)
            search_url = f"https://www.youtube.com/results?search_query={encoded_query}"
            
            # Dil parametresi ekle
            if self.lang:
                search_url += f"&hl={self.lang}"
                print(f"🌍 Arama dili: {self.lang}")
                
            print(f"🌐 URL yükleniyor: {search_url}")
            self.driver.get(search_url)

            # Sayfa yüklenmesini bekle
            video_link_selector = "a#video-title"
            try:
                WebDriverWait(self.driver, 20).until(
                    lambda driver: driver.execute_script("return document.readyState") == "complete"
                )
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, video_link_selector))
                )
                print("✅ Sayfa yüklendi")
            except TimeoutException:
                error_msg = "⏱️  Sayfa yükleme zaman aşımı"
                print(error_msg, file=sys.stderr)
                self.search_error.emit(error_msg)
                raise

            if not self._is_running:
                raise Exception("Durduruldu")

            # Video URL'lerini topla
            processed_urls = set()
            self.found_urls = []
            
            scroll_pause_time = 1.5
            scroll_distance = 3000
            last_height = self.driver.execute_script("return document.documentElement.scrollHeight")
            scroll_attempts_without_new_content = 0
            MAX_IDLE_SCROLL_ATTEMPTS = 7

            print(f"📜 Kaydırma başlıyor... Hedef: {self.limit} URL")

            while (len(self.found_urls) < self.limit and 
                   self._is_running and 
                   scroll_attempts_without_new_content < MAX_IDLE_SCROLL_ATTEMPTS):

                initial_count = len(self.found_urls)

                try:
                    video_link_elements = self.driver.find_elements(By.CSS_SELECTOR, video_link_selector)
                except Exception as e:
                    print(f"⚠️  Element bulma hatası: {e}")
                    video_link_elements = []

                for link_element in video_link_elements:
                    if len(self.found_urls) >= self.limit or not self._is_running:
                        break

                    try:
                        href = link_element.get_attribute("href")
                        if href and "/watch?v=" in href:
                            full_url = href.split('&')[0]
                            if full_url not in processed_urls:
                                self.found_urls.append(full_url)
                                processed_urls.add(full_url)
                                print(f"   ✓ Bulundu ({len(self.found_urls)}/{self.limit}): {full_url}")
                    except Exception as e:
                        continue

                if len(self.found_urls) >= self.limit or not self._is_running:
                    break

                # Aşağı kaydır
                self.driver.execute_script(f"window.scrollBy(0, {scroll_distance});")
                time.sleep(scroll_pause_time)

                new_height = self.driver.execute_script("return document.documentElement.scrollHeight")

                if new_height == last_height and len(self.found_urls) == initial_count:
                    scroll_attempts_without_new_content += 1
                    print(f"   ⏸️  Boş kaydırma: {scroll_attempts_without_new_content}/{MAX_IDLE_SCROLL_ATTEMPTS}")
                else:
                    scroll_attempts_without_new_content = 0

                last_height = new_height

            print(f"✅ Arama tamamlandı: {len(self.found_urls)} URL bulundu")

        except SessionNotCreatedException as e:
            error_msg = f"❌ ChromeDriver başlatılamadı: {e}"
            print(error_msg, file=sys.stderr)
            self.search_error.emit(error_msg)
        except Exception as e:
            error_msg = f"❌ Beklenmeyen hata: {e}"
            print(error_msg, file=sys.stderr)
            self.search_error.emit(error_msg)
        finally:
            if self.driver:
                try:
                    self.driver.quit()
                except:
                    pass
                self.driver = None

            self.search_finished.emit(self.found_urls)


# Test
if __name__ == '__main__':
    from PyQt6.QtCore import QCoreApplication
    
    app = QCoreApplication([])

    def on_finish(urls):
        print(f"\n📊 SONUÇ: {len(urls)} URL bulundu")
        for i, url in enumerate(urls, 1):
            print(f"{i}. {url}")
        app.quit()

    def on_error(msg):
        print(f"\n❌ HATA: {msg}")
        app.quit()

    worker = SearchWorker(query="python tutorial", limit=5)
    worker.search_finished.connect(on_finish)
    worker.search_error.connect(on_error)
    worker.run()