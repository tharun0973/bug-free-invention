## Reply Classification Strategy

**Q1: How would you improve a reply classifier with only 200 labeled replies?**  
To overcome limited labeled data, I’d apply data augmentation techniques:
- Paraphrasing via LLMs
- Back translation for linguistic diversity
- Synonym replacement for lexical variation

I’d also fine-tune a pre-trained transformer like DistilBERT, which generalizes well on small datasets.

---

**Q2: How would you ensure the classifier avoids biased or unsafe outputs in production?**  
- Audit training data for bias or offensive language  
- Apply filters during preprocessing  
- Use confidence thresholds and fallback logic  
- Monitor outputs with human-in-the-loop review

---

## Cold Email Generation with LLMs

**Q3: How would you design prompts for personalized cold email openers?**  
I’d use prompt engineering strategies:
- Include recipient’s industry, pain point, and product value  
- Use few-shot examples to guide tone and structure  
- Enforce constraints like length, personalization, and CTA clarity
