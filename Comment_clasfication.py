import pandas as pd
import time
import ollama
import os
from tqdm import tqdm
import json
import random
from functools import lru_cache
import colorama
from colorama import Fore, Style
import datetime
import openpyxl  # Excel desteği için


colorama.init()

CONFIG = {
    'output_folder': '',  
    'model_name': "gemma3:4b",  # 3.3GB - GPU'ya tam sığar
    'max_retries': 3,
    'retry_delay': 1,
    'cache_size': 1000 
}

def print_colored(text, color=Fore.WHITE, style=Style.NORMAL, end='\n'):
    """Renkli metin yazdır"""
    print(f"{style}{color}{text}{Style.RESET_ALL}", end=end)

def print_header(text, width=50):
    """Başlık yazdır"""
    print_colored("=" * width, Fore.CYAN, Style.BRIGHT)
    print_colored(text.center(width), Fore.CYAN, Style.BRIGHT)
    print_colored("=" * width, Fore.CYAN, Style.BRIGHT)

def print_subheader(text):
    """Alt başlık yazdır"""
    print_colored(f"\n--- {text} ---", Fore.YELLOW, Style.BRIGHT)

@lru_cache(maxsize=CONFIG['cache_size'])
def analyze_comment_for_category(comment, category, description):
    """Bir yorumu tek bir kategori için analiz et"""
    prompt = f"""
    Lütfen aşağıdaki yorumu '{category}' kategorisi için analiz et.
    
    YORUM: "{comment}"
    
    '{category}' kategorisi için sadece 1 (evet) veya 0 (hayır) olarak cevap ver.
    
    KATEGORİ AÇIKLAMASI:
    {description}
    """
    
   
    for attempt in range(CONFIG['max_retries']):
        try:
            print_colored(f"Yorum analiz ediliyor... Deneme: {attempt+1}/{CONFIG['max_retries']}", Fore.BLUE)
            
            response = ollama.chat(
                model=CONFIG['model_name'],
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.1
                }
            )
            
          
            response_text = response['message']['content'].lower()
            
           
            result = 1 if ('1' in response_text or 'evet' in response_text) else 0
            
            print_colored(f"Analiz sonucu: {result}", Fore.GREEN if result == 1 else Fore.RED)
            return result
            
        except Exception as e:
            print_colored(f"Hata: {str(e)}", Fore.RED)
            if attempt < CONFIG['max_retries'] - 1:
                print_colored(f"{CONFIG['retry_delay']} saniye bekleniyor...", Fore.YELLOW)
                time.sleep(CONFIG['retry_delay'])
   
    print_colored("Tüm denemeler başarısız oldu! Varsayılan sonuç: 0", Fore.RED, Style.BRIGHT)
    return 0

def print_category_progress(category, count, target, type_label=""):
    """Kategori ilerleme durumunu yazdır"""
    percentage = (count / target * 100) if target > 0 else 0
    progress_bar = "█" * int(percentage / 10) + "░" * (10 - int(percentage / 10))
    status = f"[{count}/{target}]"
    
    color = Fore.GREEN if type_label == "Pozitif" else Fore.RED
    
    type_text = f"{type_label} " if type_label else ""
    print_subheader(f"{category} Kategorisi {type_text}İlerlemesi")
    print_colored(f"İlerleme: {progress_bar} {status} - %{percentage:.1f}", color)
    print_colored(f"Hedef: {target} adet {type_label.lower()} eşleşme bulmak", Fore.CYAN)
    print_colored("-----------------------------", Fore.YELLOW)

def save_category_results(category, results, output_file):
    """Bir kategori için sonuçları kaydet"""
    df = pd.DataFrame(results)
    # Excel formatında kaydet
    excel_output = output_file.replace('.csv', '.xlsx')
    df.to_excel(excel_output, index=False)
    print_colored(f"Sonuçlar {excel_output} dosyasına kaydedildi.", Fore.GREEN, Style.BRIGHT)
    # CSV olarak da kaydet (yedek)
    df.to_csv(output_file, index=False)
    return excel_output

def process_category(category_name, category_description, target_positive, target_negative, df, file_type="all"):
    """Belirtilen kategori için yorumları analiz et"""
    print_header(f"KATEGORİ: {category_name}")
    print_colored(f"Açıklama: {category_description}", Fore.CYAN)
    
   
    results = []
    
    
    count_positive = 0
    count_negative = 0
    
    
    safe_category_name = "".join(c if c.isalnum() else "_" for c in category_name)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    output_file = f"{safe_category_name}_{timestamp}.csv"
    
   
    checkpoint_file = f"{safe_category_name}_checkpoint.json"
    
    processed_indices = []
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            
            if checkpoint['target_positive'] == target_positive and checkpoint['target_negative'] == target_negative and checkpoint['category'] == category_name:
                results = checkpoint['results']
                count_positive = checkpoint['count_positive']
                count_negative = checkpoint['count_negative']
                processed_indices = checkpoint.get('processed_indices', [])
                
                print_colored(f"Checkpoint yüklendi. Şimdiye kadar {count_positive}/{target_positive} adet pozitif eşleşme ve {count_negative}/{target_negative} adet negatif eşleşme bulundu.", Fore.GREEN)
                print_colored(f"{len(processed_indices)} yorum işlendi.", Fore.BLUE)
                
                
                remaining_indices = [i for i in range(len(df)) if i not in processed_indices]
            else:
                
                results = []
                count_positive = 0
                count_negative = 0
                processed_indices = []
                remaining_indices = list(range(len(df)))
        except Exception as e:
            print_colored(f"Checkpoint yükleme hatası: {str(e)}. Yeniden başlanıyor.", Fore.RED)
            results = []
            count_positive = 0
            count_negative = 0
            processed_indices = []
            remaining_indices = list(range(len(df)))
    else:
        
        results = []
        count_positive = 0
        count_negative = 0
        processed_indices = []
        remaining_indices = list(range(len(df)))
    
    
    if file_type == "all":
       
        random.shuffle(remaining_indices)
    elif file_type == "positive_only":
       
        pass
    elif file_type == "all_comments":
       
        pass
    
    
    print_category_progress(category_name, count_positive, target_positive, "Pozitif")
    print_category_progress(category_name, count_negative, target_negative, "Negatif")
    
    
    start_time = time.time()
    total_target = target_positive + target_negative if file_type != "all_comments" else len(remaining_indices)
    
    with tqdm(total=total_target, 
              desc=f"{category_name} İşleniyor", 
              colour="green") as pbar:
        
        if file_type != "all_comments":
            pbar.update(count_positive + count_negative) 
        
        for i in remaining_indices:
           
            if count_positive >= target_positive and count_negative >= target_negative and file_type != "all_comments":
                print_colored(f"\n{category_name} için hedefler tamamlandı! ({target_positive} pozitif, {target_negative} negatif eşleşme bulundu)", Fore.GREEN, Style.BRIGHT)
                break
                
            row = df.iloc[i]
            comment = row.get('Yorumlar', row.get('Yorum', row.get('yorum', row.get('text', ''))))
            
            if not isinstance(comment, str) or len(comment.strip()) == 0:
                processed_indices.append(i)
                if file_type == "all_comments":
                    pbar.update(1)
                continue
            
           
            if count_positive >= target_positive and count_negative < target_negative and file_type != "all_comments":
               
                result = analyze_comment_for_category(comment, category_name, category_description)
                if result == 1:
                   
                    processed_indices.append(i)
                    continue
            
          
            if count_negative >= target_negative and count_positive < target_positive and file_type != "all_comments":
                
                result = analyze_comment_for_category(comment, category_name, category_description)
                if result == 0:
                    
                    processed_indices.append(i)
                    continue
            
            
            if 'result' not in locals():
                result = analyze_comment_for_category(comment, category_name, category_description)
            
           
            result_dict = {
                'topic': category_name,
                'Yorum': comment
            }
            results.append(result_dict)
            
            
            processed_indices.append(i)
            
           
            if result == 1:
                count_positive += 1
                if count_positive <= target_positive and file_type != "all_comments":
                    pbar.update(1)  
                
               
                if count_positive <= target_positive or file_type == "all_comments":
                    print_colored("\nYENİ POZİTİF EŞLEŞME BULUNDU!", Fore.GREEN, Style.BRIGHT)
                    print_colored(f"Kategori: {category_name} - İlerleme: {count_positive}/{target_positive}", Fore.YELLOW)
                    print_colored(f"Yorum: {comment[:150]}..." if len(comment) > 150 else f"Yorum: {comment}", Fore.CYAN)
                    print_colored("-" * 50, Fore.YELLOW)
            else:  
                count_negative += 1
                if count_negative <= target_negative and file_type != "all_comments":
                    pbar.update(1)  
               
                if count_negative <= target_negative or file_type == "all_comments":
                    print_colored("\nYENİ NEGATİF EŞLEŞME BULUNDU!", Fore.RED, Style.BRIGHT)
                    print_colored(f"Kategori: {category_name} - İlerleme: {count_negative}/{target_negative}", Fore.YELLOW)
                    print_colored(f"Yorum: {comment[:150]}..." if len(comment) > 150 else f"Yorum: {comment}", Fore.CYAN)
                    print_colored("-" * 50, Fore.YELLOW)
            
           
            if 'result' in locals():
                del result
            
            if file_type == "all_comments":
                pbar.update(1)
            
           
            if len(results) % 5 == 0:
              
                elapsed_time = time.time() - start_time
                total_count = count_positive + count_negative
                
                if file_type != "all_comments" and total_count > 0:
                    total_target = target_positive + target_negative
                    estimated_total = (elapsed_time / total_count) * total_target
                    remaining_time = max(0, estimated_total - elapsed_time)
                else:
                    remaining_time = 0
                
                checkpoint = {
                    'category': category_name,
                    'target_positive': target_positive,
                    'target_negative': target_negative,
                    'count_positive': count_positive,
                    'count_negative': count_negative,
                    'results': results,
                    'processed_indices': processed_indices,
                    'timestamp': datetime.datetime.now().isoformat()
                }
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint, f, ensure_ascii=False)
                
                if file_type != "all_comments":
                    print_category_progress(category_name, count_positive, target_positive, "Pozitif")
                    print_category_progress(category_name, count_negative, target_negative, "Negatif")
                    
               
                    if remaining_time > 0:
                        hours, remainder = divmod(remaining_time, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        print_colored(f"Tahmini kalan süre: {int(hours)}:{int(minutes):02d}:{int(seconds):02d}", Fore.BLUE)
    

    save_category_results(category_name, results, output_file)

    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print_header("İŞLEM TAMAMLANDI")
    print_colored(f"Toplam süre: {int(hours)}:{int(minutes):02d}:{int(seconds):02d}", Fore.BLUE)
    print_colored(f"İşlenen yorum sayısı: {len(processed_indices)}", Fore.YELLOW)
    print_colored(f"Bulunan pozitif eşleşme sayısı: {count_positive}/{target_positive}", Fore.GREEN)
    print_colored(f"Bulunan negatif eşleşme sayısı: {count_negative}/{target_negative}", Fore.RED)
    

    return (count_positive >= target_positive and count_negative >= target_negative), count_positive, count_negative, output_file

def get_default_categories_and_descriptions():
    """Referans için öntanımlı kategorileri ve açıklamalarını döndürür"""
    categories = [
        "Şarkıya Dair Yorum",
        "Sanatçıya Dair Yorum", 
        "Genel Yorum"
    ]
    
    descriptions = {
        "Şarkıya Dair Yorum": """
        Şarkının sözleri, müziği, melodisi, performansı veya klip gibi esere yönelik değerlendirme içeren yorumlardır. 
        Kişi veya duygudan ziyade doğrudan şarkının kendisine odaklanır.
        Anahtar Kelimeler: şarkı, söz, melodi, müzik, beste, aranjman, klip, video, ritim, enstrüman, gitar, bağlama, ses, ton, akor, nakarat, kuple, intro, outro, cover, remix, canlı performans, stüdyo, kayıt, albüm, single, parça, eser, yapım, prodüksiyon
        """,
        "Sanatçıya Dair Yorum": """
        Müslüm Gürses'in kişiliği, sesi, kariyeri, hayatı veya ona yönelik övgü/eleştiri içeren yorumlardır. 
        Odak noktası şarkı değil sanatçının kendisidir.
        Anahtar Kelimeler: Müslüm, Müslüm Baba, sanatçı, şarkıcı, ses, yorum, kariyer, hayat, biyografi, efsane, usta, büyük, merhum, rahmetli, anma, özlem, vefa, saygı, sevgi, hayran, fan, idol, karizmatik, duruş, sahne, konser, röportaj, anı, hatıra
        """,
        "Genel Yorum": """
        Şarkı veya sanatçıya özel bir odak taşımayan, belirgin bilgi ya da detay içermeden genel görüş, düşünce veya sohbet niteliğinde yapılmış yorumlardır. 
        İçeriği geniştir ve belirli bir alt kategoriye net şekilde oturmaz.
        Anahtar Kelimeler: güzel, süper, harika, muhteşem, iyi, kötü, beğendim, sevdim, dinliyorum, tekrar, yine, hep, her zaman, canım, gönül, duygu, his, ağlamak, üzülmek, mutlu, hüzün, nostalji, eski, yeni, bugün, gece, sabah
        """
    }
    return categories, descriptions

def main():
    print_header("Yorum Kategori Analiz Programı", 60)
    
    # Sabit dosya yolu
    excel_file_path = r"C:\Users\hamdi\Desktop\youtube_comment_scraper\muslum_gurses_yorumlari_temizlenmis.xlsx"
    
    categories, descriptions = get_default_categories_and_descriptions()
    
    print_colored("\nKategoriler:", Fore.YELLOW)
    for i, category in enumerate(categories, 1):
        print_colored(f"{i}. {category}", Fore.CYAN)
    
    # Veri setini yükle
    try:
        print_colored(f"\nVeri seti yükleniyor: {excel_file_path}", Fore.BLUE)
        df = pd.read_excel(excel_file_path)
        print_colored(f"Veri seti yüklendi. Toplam {len(df)} yorum var.", Fore.GREEN)
    except Exception as e:
        print_colored(f"Excel yükleme hatası: {str(e)}", Fore.RED, Style.BRIGHT)
        return
    
    all_results = []
    used_indices = set()  # Kullanılan yorum indekslerini takip et
    
    for category_name in categories:
        category_description = descriptions[category_name]
        print_colored(f"\n{'='*60}", Fore.MAGENTA)
        print_colored(f"Kategori işleniyor: {category_name}", Fore.MAGENTA, Style.BRIGHT)
        print_colored(f"Kalan kullanılabilir yorum: {len(df) - len(used_indices)}", Fore.YELLOW)
        print_colored(f"{'='*60}", Fore.MAGENTA)
        
        # Bu kategori için sonuçları topla
        category_results = []
        count = 0
        target = 300
        
        # Kullanılmamış yorumları al ve karıştır
        available_indices = [i for i in range(len(df)) if i not in used_indices]
        random.shuffle(available_indices)
        
        for i in tqdm(available_indices, desc=f"{category_name} taranıyor"):
            if count >= target:
                break
            
            row = df.iloc[i]
            comment = row.get('Temizlenmis_Yorumlar', row.get('Yorumlar', row.get('Yorum', row.get('text', ''))))
            
            if not isinstance(comment, str) or len(comment.strip()) == 0:
                continue
            
            # Yorumu analiz et
            result = analyze_comment_for_category(comment, category_name, category_description)
            
            if result == 1:  # Eşleşme bulundu
                category_results.append({
                    'topic': category_name,
                    'Temizlenmis_Yorumlar': comment
                })
                used_indices.add(i)  # Bu yorumu kullanıldı olarak işaretle
                count += 1
                print_colored(f"\n[{count}/{target}] Eşleşme bulundu!", Fore.GREEN)
                print_colored(f"Yorum: {comment[:100]}...", Fore.CYAN)
        
        print_colored(f"\n{category_name} için {count} yorum bulundu.", Fore.GREEN, Style.BRIGHT)
        all_results.extend(category_results)
    
    # Tüm sonuçları birleştir ve kaydet
    if all_results:
        final_df = pd.DataFrame(all_results)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        final_output = f"tum_kategoriler_{timestamp}.xlsx"
        final_df.to_excel(final_output, index=False)
        print_header("TÜM KATEGORİLER TAMAMLANDI")
        print_colored(f"Toplam {len(final_df)} yorum kaydedildi.", Fore.GREEN, Style.BRIGHT)
        print_colored(f"Dosya: {final_output}", Fore.CYAN)
        
        # Kategori dağılımını göster
        print_colored("\nKategori Dağılımı:", Fore.YELLOW)
        for cat in categories:
            cat_count = len([r for r in all_results if r['topic'] == cat])
            print_colored(f"  {cat}: {cat_count} yorum", Fore.CYAN)
    
    print_colored(f"\nProgram tamamlandı. Teşekkürler!", Fore.GREEN, Style.BRIGHT)

if __name__ == "__main__":
    main()