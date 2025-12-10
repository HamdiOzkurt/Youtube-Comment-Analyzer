"""
Content Assistant - İçerik Üretici Asistanı
Yorumlardan soru, talep ve öneri çıkarma
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class ContentInsight:
    """İçerik içgörüsü"""
    category: str  # 'question', 'request', 'suggestion', 'complaint', 'praise'
    text: str
    confidence: float
    keywords: List[str] = field(default_factory=list)


@dataclass
class AudienceAnalysis:
    """Kitle analizi sonucu"""
    questions: List[ContentInsight]
    requests: List[ContentInsight]
    suggestions: List[ContentInsight]
    complaints: List[ContentInsight]
    praises: List[ContentInsight]
    summary: Dict


class ContentAssistant:
    """İçerik üretici asistanı - yorumlardan içgörü çıkarır"""
    
    # Soru kalıpları
    QUESTION_PATTERNS = [
        r'.+\?$',  # Soru işareti ile biten
        r'nasıl\s+.+',
        r'ne\s+zaman\s+.+',
        r'neden\s+.+',
        r'niye\s+.+',
        r'kim\s+.+',
        r'hangi\s+.+',
        r'kaç\s+.+',
        r'ne\s+kadar\s+.+',
        r'nerede\s+.+',
        r'.+\s+mı\??',
        r'.+\s+mi\??',
        r'.+\s+mu\??',
        r'.+\s+mü\??',
    ]
    
    # Talep kalıpları
    REQUEST_PATTERNS = [
        r'lütfen\s+.+',
        r'rica\s+.+',
        r'.+\s+yapar\s*mısın',
        r'.+\s+yapar\s*mısınız',
        r'.+\s+yapabilir\s*misin',
        r'.+\s+yapabilir\s*misiniz',
        r'.+\s+ister\s*misin',
        r'.+\s+ister\s*misiniz',
        r'.+\s+bekl[ie]yorum',
        r'.+\s+bekl[ie]yoruz',
        r'.+\s+istiyorum',
        r'.+\s+istiyoruz',
        r'devamını\s+.+',
        r'.+\s+çek\s*(?:in|siniz)',
        r'.+\s+yap\s*(?:ın|sanız)',
        r'.+\s+paylaş\s*(?:ın|sanız)',
        r'.+\s+at\s*(?:ın|sanız)',
        r'daha\s+fazla\s+.+',
    ]
    
    # Öneri kalıpları
    SUGGESTION_PATTERNS = [
        r'.+\s+olsa\s+güzel\s+olur',
        r'.+\s+olabilir',
        r'.+\s+olmalı',
        r'.+\s+yapılmalı',
        r'.+\s+daha\s+iyi\s+olur',
        r'.+\s+tavsiye\s+ederim',
        r'.+\s+öneririm',
        r'keşke\s+.+',
        r'bence\s+.+',
        r'.+\s+düşünüyorum',
    ]
    
    # Şikayet kalıpları
    COMPLAINT_PATTERNS = [
        r'.+\s+berbat',
        r'.+\s+kötü',
        r'.+\s+rezalet',
        r'.+\s+beğenmedim',
        r'.+\s+hayal\s+kırıklığı',
        r'.+\s+beklentimi\s+karşılamadı',
        r'.+\s+vakit\s+kaybı',
        r'.+\s+saçma',
        r'.+\s+anlamsız',
        r'hiç\s+.+\s+değil',
        r'.+\s+sıkıcı',
        r'.+\s+eksik',
    ]
    
    # Övgü kalıpları
    PRAISE_PATTERNS = [
        r'.+\s+harika',
        r'.+\s+muhteşem',
        r'.+\s+mükemmel',
        r'.+\s+süper',
        r'.+\s+efsane',
        r'.+\s+çok\s+güzel',
        r'.+\s+çok\s+iyi',
        r'.+\s+bayıldım',
        r'.+\s+aşık\s+oldum',
        r'.+\s+beğendim',
        r'.+\s+tebrikler',
        r'.+\s+bravo',
        r'.+\s+helal\s+olsun',
        r'10\s*numara',
        r'5\s*yıldız',
    ]
    
    def __init__(self):
        """ContentAssistant başlat"""
        self.compiled_patterns = {
            'question': [re.compile(p, re.IGNORECASE) for p in self.QUESTION_PATTERNS],
            'request': [re.compile(p, re.IGNORECASE) for p in self.REQUEST_PATTERNS],
            'suggestion': [re.compile(p, re.IGNORECASE) for p in self.SUGGESTION_PATTERNS],
            'complaint': [re.compile(p, re.IGNORECASE) for p in self.COMPLAINT_PATTERNS],
            'praise': [re.compile(p, re.IGNORECASE) for p in self.PRAISE_PATTERNS],
        }
    
    def _match_category(self, text: str, category: str) -> float:
        """Kategoriye uygunluk skoru hesapla"""
        if not text:
            return 0.0
        
        patterns = self.compiled_patterns.get(category, [])
        matches = sum(1 for p in patterns if p.search(text))
        
        return min(matches / 3, 1.0)  # Normalize (0-1)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Metinden anahtar kelimeleri çıkar"""
        # Basit keyword extraction
        words = re.findall(r'\b[a-zçğıöşü]{4,}\b', text.lower())
        # Stop words filtreleme (basit)
        stop_words = {'için', 'daha', 'çok', 'gibi', 'kadar', 'nasıl', 'olan', 'olarak'}
        return list(set(w for w in words if w not in stop_words))[:5]
    
    def classify_comment(self, text: str) -> List[ContentInsight]:
        """Yorumu sınıflandır"""
        if not text or len(text.strip()) < 5:
            return []
        
        insights = []
        text = text.strip()
        
        for category in ['question', 'request', 'suggestion', 'complaint', 'praise']:
            confidence = self._match_category(text, category)
            
            if confidence > 0.1:  # Eşik değer
                insights.append(ContentInsight(
                    category=category,
                    text=text[:200],  # Kısalt
                    confidence=confidence,
                    keywords=self._extract_keywords(text)
                ))
        
        return sorted(insights, key=lambda x: x.confidence, reverse=True)
    
    def analyze_comments(self, comments: List[str]) -> AudienceAnalysis:
        """Tüm yorumları analiz et"""
        questions = []
        requests = []
        suggestions = []
        complaints = []
        praises = []
        
        for comment in comments:
            insights = self.classify_comment(comment)
            
            for insight in insights:
                if insight.confidence >= 0.3:  # Güvenilir eşik
                    if insight.category == 'question':
                        questions.append(insight)
                    elif insight.category == 'request':
                        requests.append(insight)
                    elif insight.category == 'suggestion':
                        suggestions.append(insight)
                    elif insight.category == 'complaint':
                        complaints.append(insight)
                    elif insight.category == 'praise':
                        praises.append(insight)
        
        # Güvenilirlik skoruna göre sırala
        questions.sort(key=lambda x: x.confidence, reverse=True)
        requests.sort(key=lambda x: x.confidence, reverse=True)
        suggestions.sort(key=lambda x: x.confidence, reverse=True)
        complaints.sort(key=lambda x: x.confidence, reverse=True)
        praises.sort(key=lambda x: x.confidence, reverse=True)
        
        total = len(comments)
        
        return AudienceAnalysis(
            questions=questions[:20],  # Top 20
            requests=requests[:20],
            suggestions=suggestions[:20],
            complaints=complaints[:20],
            praises=praises[:20],
            summary={
                'total_comments': total,
                'question_count': len(questions),
                'request_count': len(requests),
                'suggestion_count': len(suggestions),
                'complaint_count': len(complaints),
                'praise_count': len(praises),
                'question_ratio': len(questions) / total if total > 0 else 0,
                'request_ratio': len(requests) / total if total > 0 else 0,
                'positive_ratio': len(praises) / total if total > 0 else 0,
                'negative_ratio': len(complaints) / total if total > 0 else 0,
            }
        )
    
    def get_questions(self, comments: List[str]) -> List[str]:
        """Sadece soruları çıkar"""
        questions = []
        
        for comment in comments:
            if '?' in comment:
                # Soru cümlelerini ayır
                sentences = re.split(r'[.!]', comment)
                for sent in sentences:
                    if '?' in sent:
                        q = sent.strip()
                        if len(q) > 10:
                            questions.append(q)
        
        return list(set(questions))  # Tekrarları kaldır
    
    def get_requests(self, comments: List[str]) -> List[str]:
        """Sadece talepleri çıkar"""
        requests = []
        
        for comment in comments:
            for pattern in self.compiled_patterns['request']:
                matches = pattern.findall(comment.lower())
                requests.extend(matches)
        
        return list(set(requests))[:50]  # Top 50
    
    def get_content_ideas(self, analysis: AudienceAnalysis) -> List[str]:
        """Analiz sonucundan içerik fikirleri çıkar"""
        ideas = []
        
        # Sorulardan fikir çıkar
        if analysis.questions:
            ideas.append("❓ En çok sorulan sorular için Q&A videosu")
        
        # Taleplerden fikir çıkar
        if analysis.requests:
            top_requests = analysis.requests[:3]
            for req in top_requests:
                ideas.append(f"📢 Talep: {req.text[:50]}...")
        
        # Önerilerden fikir çıkar
        if analysis.suggestions:
            ideas.append("💡 İzleyici önerileri dikkate alınabilir")
        
        # Şikayetlerden fikir çıkar
        if analysis.complaints:
            ideas.append("⚠️ Şikayet edilen konular iyileştirilebilir")
        
        return ideas


# ============= TEST KODU =============
if __name__ == '__main__':
    print("=" * 60)
    print("🤖 CONTENT ASSISTANT TEST")
    print("=" * 60)
    
    assistant = ContentAssistant()
    
    test_comments = [
        "Bu şarkı harika olmuş, çok beğendim!",
        "Bir sonraki video ne zaman gelecek?",
        "Lütfen daha fazla rock müzik yapın",
        "Klip biraz sıkıcı olmuş, beklentimi karşılamadı",
        "Vokal mükemmel, tebrikler!",
        "Akustik versiyon olsa çok güzel olur",
        "Neden bu kadar kısa tutmuşsunuz videoyu?",
        "Devamını sabırsızlıkla bekliyorum",
        "Berbat olmuş, hiç beğenmedim",
        "10 numara! Efsane!",
        "Canlı performans videosu çeker misiniz?",
        "Bence daha enerjik şarkılar yapmalısınız",
    ]
    
    analysis = assistant.analyze_comments(test_comments)
    
    print("\n📋 ANALİZ SONUÇLARI:\n")
    
    print(f"📊 Toplam Yorum: {analysis.summary['total_comments']}")
    print(f"❓ Sorular: {analysis.summary['question_count']}")
    print(f"📢 Talepler: {analysis.summary['request_count']}")
    print(f"💡 Öneriler: {analysis.summary['suggestion_count']}")
    print(f"⚠️ Şikayetler: {analysis.summary['complaint_count']}")
    print(f"👏 Övgüler: {analysis.summary['praise_count']}")
    
    if analysis.questions:
        print("\n❓ ÖNE ÇIKAN SORULAR:")
        for q in analysis.questions[:3]:
            print(f"   • {q.text}")
    
    if analysis.requests:
        print("\n📢 ÖNE ÇIKAN TALEPLER:")
        for r in analysis.requests[:3]:
            print(f"   • {r.text}")
    
    print("\n💡 İÇERİK FİKİRLERİ:")
    for idea in assistant.get_content_ideas(analysis):
        print(f"   {idea}")
