"""
Türkçe BERT Tabanlı Duygu Analiz Motoru (Promptsuz Yerel NLP)

Bu modül, BIST 100 hisselerine ait KAP açıklamaları ve finansal haber başlıklarını
Hugging Face üzerindeki yerel Türkçe BERT modeli (savasy/bert-base-turkish-sentiment-cased)
ve PyTorch GPU (CUDA) altyapısı ile yüksek hızda (milisaniyeler mertebesinde) skorlar.
Sıfır LLM API maliyeti, sıfır kota riski ve tam deterministik sürekli skor [-1.0, +1.0] üretir.
"""

import sys
from typing import List, Dict, Any, Union
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEFAULT_MODEL_NAME = "savasy/bert-base-turkish-sentiment-cased"


class TurkishBertSentimentEngine:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[BERT Engine] Model yükleniyor: '{model_name}' (Cihaz: {self.device})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # GPU'da FP16 ile inferans hızını ve bellek verimini 2 katına çıkar
        if self.device.type == "cuda":
            self.model = self.model.half().to(self.device)
        else:
            self.model = self.model.to(self.device)

        self.model.eval()

        # Etiket haritalama (id2label analizi)
        self.id2label = getattr(self.model.config, "id2label", {0: "negative", 1: "positive"})
        self.pos_idx = 1
        self.neg_idx = 0
        
        for idx, lbl in self.id2label.items():
            l_str = str(lbl).lower()
            if "pos" in l_str or "olumlu" in l_str:
                self.pos_idx = int(idx)
            elif "neg" in l_str or "olumsuz" in l_str:
                self.neg_idx = int(idx)

        print(f"[BERT Engine] Model hazır. Pozitif Sınıf İndeksi: {self.pos_idx}, Negatif Sınıf İndeksi: {self.neg_idx}")

    @torch.inference_mode()
    def score_texts(self, texts: List[str], batch_size: int = 64) -> List[Dict[str, Any]]:
        """
        Metin listesi için toplu çıkarım yapar.
        Dönen her kayıt: {'text': str, 'score': float (-1.0 ile +1.0 arası), 'pos_prob': float, 'neg_prob': float, 'label': str}
        """
        if not texts:
            return []

        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            clean_batch = [t.strip() if isinstance(t, str) and t.strip() else "nötr bilgi" for t in batch_texts]

            inputs = self.tokenizer(
                clean_batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits.float(), dim=-1).cpu().numpy()

            for text, prob in zip(batch_texts, probs):
                if not text or not text.strip():
                    results.append({
                        "text": text,
                        "score": 0.0,
                        "pos_prob": 0.5,
                        "neg_prob": 0.5,
                        "label": "NOTR"
                    })
                    continue

                pos_p = float(prob[self.pos_idx])
                neg_p = float(prob[self.neg_idx])
                score = round(pos_p - neg_p, 4)

                if score > 0.15:
                    label = "POZITIF"
                elif score < -0.15:
                    label = "NEGATIF"
                else:
                    label = "NOTR"

                results.append({
                    "text": text,
                    "score": score,
                    "pos_prob": round(pos_p, 4),
                    "neg_prob": round(neg_p, 4),
                    "label": label
                })

        return results

    def score_single(self, text: str) -> float:
        """Tek bir metin için [-1.0, +1.0] duygu skorunu döner."""
        res = self.score_texts([text])
        return res[0]["score"] if res else 0.0


# Global singleton instance (tekrar tekrar yükleme maliyetini engellemek için)
_GLOBAL_ENGINE = None

def get_sentiment_engine() -> TurkishBertSentimentEngine:
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = TurkishBertSentimentEngine()
    return _GLOBAL_ENGINE


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    engine = get_sentiment_engine()
    test_samples = [
        "Şirketimiz 50 milyon USD bedelle dev batarya tesisi ihracat sözleşmesi imzalamıştır.",
        "Vergi dairesi tarafından şirketimize 850 milyon TL usulsüzlük cezası tebliğ edildi.",
        "Şirketimizin olağan genel kurul toplantısı İstanbul merkezinde başarıyla tamamlanmıştır.",
        "Şirket üçüncü çeyrekte beklentilerin %45 üzerinde net kâr açıkladı.",
        "Fabrikada çıkan yangın sebebiyle üretim faaliyetleri belirsiz bir süre için durdurulmuştur.",
        "BISTECH pay bazında devre kesici uygulaması bildirimi."
    ]

    print("\n" + "=" * 80)
    print("TÜRKÇE BERT DUYGU SKORLAMA DOĞRULAMA TESTİ")
    print("=" * 80)
    scored = engine.score_texts(test_samples)
    for s in scored:
        print(f"[{s['label']:<7}] Skor: {s['score']:>+6.3f} (Pos: {s['pos_prob']:.2f}, Neg: {s['neg_prob']:.2f}) | {s['text']}")
    print("=" * 80)
