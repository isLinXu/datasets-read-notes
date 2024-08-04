# datasets-read-notes

datasets-read-notes


- multimodal datasets
https://huggingface.co/datasets?sort=likes&search=multimodal

# Awesome Datasets

## Datasets of Pre-Training for Alignment

| Name                    | Paper                                                        | Type    | Modalities              |
| ----------------------- | ------------------------------------------------------------ | ------- | ----------------------- |
| **ShareGPT4Video**      | [ShareGPT4Video: Improving Video Understanding and Generation with Better Captions](https://arxiv.org/pdf/2406.04325v1) | Caption | Video-Text              |
| **COYO-700M**           | [COYO-700M: Image-Text Pair Dataset](https://github.com/kakaobrain/coyo-dataset/) | Caption | Image-Text              |
| **ShareGPT4V**          | [ShareGPT4V: Improving Large Multi-Modal Models with Better Captions](https://arxiv.org/pdf/2311.12793.pdf) | Caption | Image-Text              |
| **AS-1B**               | [The All-Seeing Project: Towards Panoptic Visual Recognition and Understanding of the Open World](https://arxiv.org/pdf/2308.01907.pdf) | Hybrid  | Image-Text              |
| **InternVid**           | [InternVid: A Large-scale Video-Text Dataset for Multimodal Understanding and Generation](https://arxiv.org/pdf/2307.06942.pdf) | Caption | Video-Text              |
| **MS-COCO**             | [Microsoft COCO: Common Objects in Context](https://arxiv.org/pdf/1405.0312.pdf) | Caption | Image-Text              |
| **SBU Captions**        | [Im2Text: Describing Images Using 1 Million Captioned Photographs](https://proceedings.neurips.cc/paper/2011/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf) | Caption | Image-Text              |
| **Conceptual Captions** | [Conceptual Captions: A Cleaned, Hypernymed, Image Alt-text Dataset For Automatic Image Captioning](https://aclanthology.org/P18-1238.pdf) | Caption | Image-Text              |
| **LAION-400M**          | [LAION-400M: Open Dataset of CLIP-Filtered 400 Million Image-Text Pairs](https://arxiv.org/pdf/2111.02114.pdf) | Caption | Image-Text              |
| **VG Captions**         | [Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations](https://link.springer.com/content/pdf/10.1007/s11263-016-0981-7.pdf) | Caption | Image-Text              |
| **Flickr30k**           | [Flickr30k Entities: Collecting Region-to-Phrase Correspondences for Richer Image-to-Sentence Models](https://openaccess.thecvf.com/content_iccv_2015/papers/Plummer_Flickr30k_Entities_Collecting_ICCV_2015_paper.pdf) | Caption | Image-Text              |
| **AI-Caps**             | [AI Challenger : A Large-scale Dataset for Going Deeper in Image Understanding](https://arxiv.org/pdf/1711.06475.pdf) | Caption | Image-Text              |
| **Wukong Captions**     | [Wukong: A 100 Million Large-scale Chinese Cross-modal Pre-training Benchmark](https://proceedings.neurips.cc/paper_files/paper/2022/file/a90b9a09a6ee43d6631cf42e225d73b4-Paper-Datasets_and_Benchmarks.pdf) | Caption | Image-Text              |
| **GRIT**                | [Kosmos-2: Grounding Multimodal Large Language Models to the World](https://arxiv.org/pdf/2306.14824.pdf) | Caption | Image-Text-Bounding-Box |
| **Youku-mPLUG**         | [Youku-mPLUG: A 10 Million Large-scale Chinese Video-Language Dataset for Pre-training and Benchmarks](https://arxiv.org/pdf/2306.04362.pdf) | Caption | Video-Text              |
| **MSR-VTT**             | [MSR-VTT: A Large Video Description Dataset for Bridging Video and Language](https://openaccess.thecvf.com/content_cvpr_2016/papers/Xu_MSR-VTT_A_Large_CVPR_2016_paper.pdf) | Caption | Video-Text              |
| **Webvid10M**           | [Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval](https://arxiv.org/pdf/2104.00650.pdf) | Caption | Video-Text              |
| **WavCaps**             | [WavCaps: A ChatGPT-Assisted Weakly-Labelled Audio Captioning Dataset for Audio-Language Multimodal Research](https://arxiv.org/pdf/2303.17395.pdf) | Caption | Audio-Text              |
| **AISHELL-1**           | [AISHELL-1: An open-source Mandarin speech corpus and a speech recognition baseline](https://arxiv.org/pdf/1709.05522.pdf) | ASR     | Audio-Text              |
| **AISHELL-2**           | [AISHELL-2: Transforming Mandarin ASR Research Into Industrial Scale](https://arxiv.org/pdf/1808.10583.pdf) | ASR     | Audio-Text              |
| **VSDial-CN**           | [X-LLM: Bootstrapping Advanced Large Language Models by Treating Multi-Modalities as Foreign Languages](https://arxiv.org/pdf/2305.04160.pdf) | ASR     | Image-Audio-Text        |

## Datasets of Multimodal Instruction Tuning

| Name                    | Paper                                                        | Link                                                         | Notes                                                        |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **VEGA**                | [VEGA: Learning Interleaved Image-Text Comprehension in Vision-Language Large Models](https://arxiv.org/pdf/2406.10228) | [Link](https://github.com/zhourax/VEGA)                      | A dataset for enchancing model capabilities in comprehension of interleaved information |
| **ALLaVA-4V**           | [ALLaVA: Harnessing GPT4V-synthesized Data for A Lite Vision-Language Model](https://arxiv.org/pdf/2402.11684.pdf) | [Link](https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V) | Vision and language caption and instruction dataset generated by GPT4V |
| **IDK**                 | [Visually Dehallucinative Instruction Generation: Know What You Don't Know](https://arxiv.org/pdf/2402.09717.pdf) | [Link](https://github.com/ncsoft/idk)                        | Dehallucinative visual instruction for "I Know" hallucination |
| **CAP2QA**              | [Visually Dehallucinative Instruction Generation](https://arxiv.org/pdf/2402.08348.pdf) | [Link](https://github.com/ncsoft/cap2qa)                     | Image-aligned visual instruction dataset                     |
| **M3DBench**            | [M3DBench: Let's Instruct Large Models with Multi-modal 3D Prompts](https://arxiv.org/pdf/2312.10763.pdf) | [Link](https://github.com/OpenM3D/M3DBench)                  | A large-scale 3D instruction tuning dataset                  |
| **ViP-LLaVA-Instruct**  | [Making Large Multimodal Models Understand Arbitrary Visual Prompts](https://arxiv.org/pdf/2312.00784.pdf) | [Link](https://huggingface.co/datasets/mucai/ViP-LLaVA-Instruct) | A mixture of LLaVA-1.5 instruction data and the region-level visual prompting data |
| **LVIS-Instruct4V**     | [To See is to Believe: Prompting GPT-4V for Better Visual Instruction Tuning](https://arxiv.org/pdf/2311.07574.pdf) | [Link](https://huggingface.co/datasets/X2FD/LVIS-Instruct4V) | A visual instruction dataset via self-instruction from GPT-4V |
| **ComVint**             | [What Makes for Good Visual Instructions? Synthesizing Complex Visual Reasoning Instructions for Visual Instruction Tuning](https://arxiv.org/pdf/2311.01487.pdf) | [Link](https://github.com/RUCAIBox/ComVint#comvint-data)     | A synthetic instruction dataset for complex visual reasoning |
| **SparklesDialogue**    | [✨Sparkles: Unlocking Chats Across Multiple Images for Multimodal Instruction-Following Models](https://arxiv.org/pdf/2308.16463.pdf) | [Link](https://github.com/HYPJUDY/Sparkles#sparklesdialogue) | A machine-generated dialogue dataset tailored for word-level interleaved multi-image and text interactions to augment the conversational competence of instruction-following LLMs across multiple images and dialogue turns. |
| **StableLLaVA**         | [StableLLaVA: Enhanced Visual Instruction Tuning with Synthesized Image-Dialogue Data](https://arxiv.org/pdf/2308.10253v1.pdf) | [Link](https://github.com/icoz69/StableLLAVA)                | A cheap and effective approach to collect visual instruction tuning data |
| **M-HalDetect**         | [Detecting and Preventing Hallucinations in Large Vision Language Models](https://arxiv.org/pdf/2308.06394.pdf) | [Coming soon](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/blob/main) | A dataset used to train and benchmark models for hallucination detection and prevention |
| **MGVLID**              | [ChatSpot: Bootstrapping Multimodal LLMs via Precise Referring Instruction Tuning](https://arxiv.org/pdf/2307.09474.pdf) | -                                                            | A high-quality instruction-tuning dataset including image-text and region-text pairs |
| **BuboGPT**             | [BuboGPT: Enabling Visual Grounding in Multi-Modal LLMs](https://arxiv.org/pdf/2307.08581.pdf) | [Link](https://huggingface.co/datasets/magicr/BuboGPT)       | A high-quality instruction-tuning dataset including audio-text audio caption data and audio-image-text localization data |
| **SVIT**                | [SVIT: Scaling up Visual Instruction Tuning](https://arxiv.org/pdf/2307.04087.pdf) | [Link](https://huggingface.co/datasets/BAAI/SVIT)            | A large-scale dataset with 4.2M informative visual instruction tuning data, including conversations, detailed descriptions, complex reasoning and referring QAs |
| **mPLUG-DocOwl**        | [mPLUG-DocOwl: Modularized Multimodal Large Language Model for Document Understanding](https://arxiv.org/pdf/2307.02499.pdf) | [Link](https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/DocLLM) | An instruction tuning dataset featuring a wide range of visual-text understanding tasks including OCR-free document understanding |
| **PF-1M**               | [Visual Instruction Tuning with Polite Flamingo](https://arxiv.org/pdf/2307.01003.pdf) | [Link](https://huggingface.co/datasets/chendelong/PF-1M/tree/main) | A collection of 37 vision-language datasets with responses rewritten by Polite Flamingo. |
| **ChartLlama**          | [ChartLlama: A Multimodal LLM for Chart Understanding and Generation](https://arxiv.org/pdf/2311.16483.pdf) | [Link](https://huggingface.co/datasets/listen2you002/ChartLlama-Dataset) | A multi-modal instruction-tuning dataset for chart understanding and generation |
| **LLaVAR**              | [LLaVAR: Enhanced Visual Instruction Tuning for Text-Rich Image Understanding](https://arxiv.org/pdf/2306.17107.pdf) | [Link](https://llavar.github.io/#data)                       | A visual instruction-tuning dataset for Text-rich Image Understanding |
| **MotionGPT**           | [MotionGPT: Human Motion as a Foreign Language](https://arxiv.org/pdf/2306.14795.pdf) | [Link](https://github.com/OpenMotionLab/MotionGPT)           | A instruction-tuning dataset including multiple human motion-related tasks |
| **LRV-Instruction**     | [Mitigating Hallucination in Large Multi-Modal Models via Robust Instruction Tuning](https://arxiv.org/pdf/2306.14565.pdf) | [Link](https://github.com/FuxiaoLiu/LRV-Instruction#visual-instruction-data-lrv-instruction) | Visual instruction tuning dataset for addressing hallucination issue |
| **Macaw-LLM**           | [Macaw-LLM: Multi-Modal Language Modeling with Image, Audio, Video, and Text Integration](https://arxiv.org/pdf/2306.09093.pdf) | [Link](https://github.com/lyuchenyang/Macaw-LLM/tree/main/data) | A large-scale multi-modal instruction dataset in terms of multi-turn dialogue |
| **LAMM-Dataset**        | [LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark](https://arxiv.org/pdf/2306.06687.pdf) | [Link](https://github.com/OpenLAMM/LAMM#lamm-dataset)        | A comprehensive multi-modal instruction tuning dataset       |
| **Video-ChatGPT**       | [Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models](https://arxiv.org/pdf/2306.05424.pdf) | [Link](https://github.com/mbzuai-oryx/Video-ChatGPT#video-instruction-dataset-open_file_folder) | 100K high-quality video instruction dataset                  |
| **MIMIC-IT**            | [MIMIC-IT: Multi-Modal In-Context Instruction Tuning](https://arxiv.org/pdf/2306.05425.pdf) | [Link](https://github.com/Luodian/Otter/blob/main/mimic-it/README.md) | Multimodal in-context instruction tuning                     |
| **M3IT**                | [M3IT: A Large-Scale Dataset towards Multi-Modal Multilingual Instruction Tuning](https://arxiv.org/pdf/2306.04387.pdf) | [Link](https://huggingface.co/datasets/MMInstruction/M3IT)   | Large-scale, broad-coverage multimodal instruction tuning dataset |
| **LLaVA-Med**           | [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day](https://arxiv.org/pdf/2306.00890.pdf) | [Coming soon](https://github.com/microsoft/LLaVA-Med#llava-med-dataset) | A large-scale, broad-coverage biomedical instruction-following dataset |
| **GPT4Tools**           | [GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction](https://arxiv.org/pdf/2305.18752.pdf) | [Link](https://github.com/StevenGrove/GPT4Tools#dataset)     | Tool-related instruction datasets                            |
| **MULTIS**              | [ChatBridge: Bridging Modalities with Large Language Model as a Language Catalyst](https://arxiv.org/pdf/2305.16103.pdf) | [Coming soon](https://iva-chatbridge.github.io/)             | Multimodal instruction tuning dataset covering 16 multimodal tasks |
| **DetGPT**              | [DetGPT: Detect What You Need via Reasoning](https://arxiv.org/pdf/2305.14167.pdf) | [Link](https://github.com/OptimalScale/DetGPT/tree/main/dataset) | Instruction-tuning dataset with 5000 images and around 30000 query-answer pairs |
| **PMC-VQA**             | [PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering](https://arxiv.org/pdf/2305.10415.pdf) | [Coming soon](https://xiaoman-zhang.github.io/PMC-VQA/)      | Large-scale medical visual question-answering dataset        |
| **VideoChat**           | [VideoChat: Chat-Centric Video Understanding](https://arxiv.org/pdf/2305.06355.pdf) | [Link](https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data) | Video-centric multimodal instruction dataset                 |
| **X-LLM**               | [X-LLM: Bootstrapping Advanced Large Language Models by Treating Multi-Modalities as Foreign Languages](https://arxiv.org/pdf/2305.04160.pdf) | [Link](https://github.com/phellonchen/X-LLM)                 | Chinese multimodal instruction dataset                       |
| **LMEye**               | [LMEye: An Interactive Perception Network for Large Language Models](https://arxiv.org/pdf/2305.03701.pdf) | [Link](https://huggingface.co/datasets/YunxinLi/Multimodal_Insturction_Data_V2) | A multi-modal instruction-tuning dataset                     |
| **cc-sbu-align**        | [MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models](https://arxiv.org/pdf/2304.10592.pdf) | [Link](https://huggingface.co/datasets/Vision-CAIR/cc_sbu_align) | Multimodal aligned dataset for improving model's usability and generation's fluency |
| **LLaVA-Instruct-150K** | [Visual Instruction Tuning](https://arxiv.org/pdf/2304.08485.pdf) | [Link](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) | Multimodal instruction-following data generated by GPT       |
| **MultiInstruct**       | [MultiInstruct: Improving Multi-Modal Zero-Shot Learning via Instruction Tuning](https://arxiv.org/pdf/2212.10773.pdf) | [Link](https://github.com/VT-NLP/MultiInstruct)              | The first multimodal instruction tuning benchmark dataset    |

## Datasets of In-Context Learning

| Name         | Paper                                                        | Link                                                         | Notes                                                        |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **MIC**      | [MMICL: Empowering Vision-language Model with Multi-Modal In-Context Learning](https://arxiv.org/pdf/2309.07915.pdf) | [Link](https://huggingface.co/datasets/BleachNick/MIC_full)  | A manually constructed instruction tuning dataset including interleaved text-image inputs, inter-related multiple image inputs, and multimodal in-context learning inputs. |
| **MIMIC-IT** | [MIMIC-IT: Multi-Modal In-Context Instruction Tuning](https://arxiv.org/pdf/2306.05425.pdf) | [Link](https://github.com/Luodian/Otter/blob/main/mimic-it/README.md) | Multimodal in-context instruction dataset                    |

## Datasets of Multimodal Chain-of-Thought

| Name          | Paper                                                        | Link                                                         | Notes                                                        |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **EMER**      | [Explainable Multimodal Emotion Reasoning](https://arxiv.org/pdf/2306.15401.pdf) | [Coming soon](https://github.com/zeroQiaoba/Explainable-Multimodal-Emotion-Reasoning) | A benchmark dataset for explainable emotion reasoning task   |
| **EgoCOT**    | [EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought](https://arxiv.org/pdf/2305.15021.pdf) | [Coming soon](https://github.com/EmbodiedGPT/EmbodiedGPT_Pytorch) | Large-scale embodied planning dataset                        |
| **VIP**       | [Let’s Think Frame by Frame: Evaluating Video Chain of Thought with Video Infilling and Prediction](https://arxiv.org/pdf/2305.13903.pdf) | [Coming soon](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/blob/main) | An inference-time dataset that can be used to evaluate VideoCOT |
| **ScienceQA** | [Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering](https://proceedings.neurips.cc/paper_files/paper/2022/file/11332b6b6cf4485b84afadb1352d3a9a-Paper-Conference.pdf) | [Link](https://github.com/lupantech/ScienceQA#ghost-download-the-dataset) | Large-scale multi-choice dataset, featuring multimodal science questions and diverse domains |

## Datasets of Multimodal RLHF

| Name           | Paper                                                        | Link                                                         | Notes                                              |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------------------- |
| **VLFeedback** | [Silkie: Preference Distillation for Large Visual Language Models](https://arxiv.org/pdf/2312.10665.pdf) | [Link](https://huggingface.co/datasets/MMInstruction/VLFeedback) | A vision-language feedback dataset annotated by AI |





# Multimodal datasets for NLP Applications

1. **Sentiment Analysis**

| **Dataset**  | **Title of the Paper**                                       | **Link of the Paper**                                        | **Link of the Dataset**                                      |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| EmoDB        | A Database of German Emotional Speech                        | [Paper](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.130.8506&rep=rep1&type=pdf) | [Dataset](https://www.kaggle.com/piyushagni5/berlin-database-of-emotional-speech-emodb) |
| VAM          | The Vera am Mittag German Audio-Visual Emotional Speech Database | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4607572) | [Dataset](https://sail.usc.edu/VAM/vam_release.htm)          |
| IEMOCAP      | IEMOCAP: interactive emotional dyadic motion capture database | [Paper](https://link.springer.com/content/pdf/10.1007/s10579-008-9076-6.pdf) | [Dataset](https://sail.usc.edu/software/databases/)          |
| Mimicry      | A Multimodal Database for Mimicry Analysis                   | [Paper](https://ibug.doc.ic.ac.uk/media/uploads/documents/sun2011multimodal.pdf) | [Dataset](http://www.mahnob-db.eu/mimicry)                   |
| YouTube      | Towards Multimodal Sentiment Analysis:Harvesting Opinions from the Web | [Paper](https://ict.usc.edu/pubs/Towards Multimodal Sentiment Analysis- Harvesting Opinions from The Web.pdf) | [Dataset](https://github.com/A2Zadeh/CMU-MultimodalSDK)      |
| HUMAINE      | The HUMAINE database                                         | [Paper](https://ibug.doc.ic.ac.uk/media/uploads/documents/sun2011multimodal.pdf) | [Dataset](https://github.com/drmuskangarg/Multimodal-datasets/blob/main/www.emotion-research.net) |
| Large Movies | Sentiment classification on Large Movie Review               | [Paper](https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf) | [Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)    |
| SEMAINE      | The SEMAINE Database: Annotated Multimodal Records of Emotionally Colored Conversations between a Person and a Limited Agent | [Paper](https://ieeexplore.ieee.org/document/5959155)        | [Dataset](https://semaine-db.eu/)                            |
| AFEW         | Collecting Large, Richly Annotated Facial-Expression Databases from Movies | [Paper](http://users.cecs.anu.edu.au/~adhall/Dhall_Goecke_Lucey_Gedeon_M_2012.pdf) | [Dataset](https://cs.anu.edu.au/few/AFEW.html)               |
| SST          | Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank | [Paper](https://aclanthology.org/D13-1170.pdf)               | [Dataset](https://metatext.io/datasets/the-stanford-sentiment-treebank-(sst)) |
| ICT-MMMO     | YouTube Movie Reviews: Sentiment Analysis in an AudioVisual Context | [Paper](http://multicomp.cs.cmu.edu/wp-content/uploads/2017/09/2013_IEEEIS_wollmer_youtube.pdf) | [Dataset](https://github.com/A2Zadeh/CMU-MultimodalSDK)      |
| RECOLA       | Introducing the RECOLA multimodal corpus of remote collaborative and affective interactions | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6553805) | [Dataset](https://diuf.unifr.ch/main/diva/recola/download.html) |
| MOUD         | Utterance-Level Multimodal Sentiment Analysis                | [Paper](https://aclanthology.org/P13-1096.pdf)               |                                                              |
| CMU-MOSI     | MOSI: Multimodal Corpus of Sentiment Intensity and Subjectivity Analysis in Online Opinion Videos | [Paper](https://arxiv.org/ftp/arxiv/papers/1606/1606.06259.pdf) | [Dataset](https://github.com/A2Zadeh/CMU-MultimodalSDK)      |
| POM          | Multimodal Analysis and Prediction of Persuasiveness in Online Social Multimedia | [Paper](https://dl.acm.org/doi/pdf/10.1145/2897739)          | [Dataset](https://github.com/eusip/POM)                      |
| MELD         | MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations | [Paper](https://arxiv.org/pdf/1810.02508.pdf)                | [Dataset](https://affective-meld.github.io/)                 |
| CMU-MOSEI    | Multimodal Language Analysis in the Wild: CMU-MOSEI Dataset and Interpretable Dynamic Fusion Graph | [Paper](https://aclanthology.org/P18-1208.pdf)               | [Dataset](https://github.com/A2Zadeh/CMU-MultimodalSDK)      |
| AMMER        | Towards Multimodal Emotion Recognition in German Speech Events in Cars using Transfer Learning | [Paper](https://arxiv.org/pdf/1909.02764.pdf)                | On Request                                                   |
| SEWA         | SEWA DB: A Rich Database for Audio-Visual Emotion and Sentiment Research in the Wild | [Paper](https://arxiv.org/pdf/1901.02839.pdf)                | [Dataset](http://www.sewaproject.eu/resources)               |
| Fakeddit     | r/fakeddit: A new multimodal benchmark dataset for fine-grained fake news detection | [Paper](https://arxiv.org/pdf/1911.03854.pdf)                | [Dataset](https://fakeddit.netlify.app/)                     |
| CMU-MOSEAS   | CMU-MOSEAS: A Multimodal Language Dataset for Spanish, Portuguese, German and French | [Paper](https://aclanthology.org/2020.emnlp-main.141.pdf)    | [Dataset](https://bit.ly/2Svbg9f)                            |
| MultiOFF     | Multimodal meme dataset (MultiOFF) for identifying offensive content in image and text | [Paper](https://aclanthology.org/2020.trac-1.6.pdf)          | [Dataset](https://github.com/bharathichezhiyan/Multimodal-Meme-Classification-Identifying-Offensive-Content-in-Image-and-Text) |
| MEISD        | MEISD: A Multimodal Multi-Label Emotion, Intensity and Sentiment Dialogue Dataset for Emotion Recognition and Sentiment Analysis in Conversations | [Paper](https://aclanthology.org/2020.coling-main.393.pdf)   | [Dataset](https://github.com/declare-lab/MELD)               |
| TASS         | Overview of TASS 2020: Introducing Emotion                   | [Paper](http://ceur-ws.org/Vol-2664/tass_overview.pdf)       | [Dataset](http://www.sepln.org/workshops/tass/tass_data/download.php) |
| CH SIMS      | CH-SIMS: A Chinese Multimodal Sentiment Analysis Dataset with Fine-grained Annotation of Modality | [Paper](https://aclanthology.org/2020.acl-main.343.pdf)      | [Dataset](https://github.com/thuiar/MMSA)                    |
| Creep-Image  | A Multimodal Dataset of Images and Text                      | [Paper](http://ceur-ws.org/Vol-2769/paper_11.pdf)            | [Dataset](https://github.com/dhfbk/creep-image-dataset)      |
| Entheos      | Entheos: A Multimodal Dataset for Studying Enthusiasm        | [Paper](https://aclanthology.org/2021.findings-acl.180.pdf)  | [Dataset](https://github.com/clviegas/Entheos-Dataset)       |

1. **Machine Translation**

| **Dataset**             | **Title of the Paper**                                       | **Link of the Paper**                          | **Link of the Dataset**                                      |
| ----------------------- | ------------------------------------------------------------ | ---------------------------------------------- | ------------------------------------------------------------ |
| Multi30K                | Multi30K: Multilingual English-German Image Description      | [Paper](https://arxiv.org/pdf/1605.00459.pdf)  | [Dataset](https://github.com/multi30k/dataset)               |
| How2                    | How2: A Large-scale Dataset for Multimodal Language Understanding | [Paper](https://arxiv.org/pdf/1811.00347.pdf)  | [Dataset](https://github.com/srvk/how2-dataset)              |
| MLT                     | Multimodal Lexical Translation                               | [Paper](https://aclanthology.org/L18-1602.pdf) | [Dataset](https://github.com/sheffieldnlp/mlt)               |
| IKEA                    | A Visual Attention Grounding Neural Model for Multimodal Machine Translation | [Paper](https://arxiv.org/pdf/1808.08266.pdf)  | [Dataset](https://github.com/sampalomad/IKEA-Dataset)        |
| Flickr30K (EN- (hi-IN)) | Multimodal Neural Machine Translation for Low-resource Language Pairs using Synthetic Data | [Paper](https://aclanthology.org/W18-3405.pdf) | On Request                                                   |
| Hindi Visual Genome     | Hindi Visual Genome: A Dataset for Multimodal English-to-Hindi Machine Translation | [Paper](https://arxiv.org/pdf/1907.08948.pdf)  | [Dataset](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2997) |
| HowTo100M               | Multilingual Multimodal Pre-training for Zero-Shot Cross-Lingual Transfer of Vision-Language Models | [Paper](https://arxiv.org/pdf/2103.08849.pdf)  | [Dataset](https://github.com/berniebear/Multi-HT100M)        |

1. **Information Retrieval**

| **Dataset**          | **Title of the Paper**                                       | **Link of the Paper**                                        | **Link of the Dataset**                                      |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| MUSICLEF             | MusiCLEF: a Benchmark Activity in Multimodal Music Information Retrieval | [Paper](https://ismir2011.ismir.net/papers/OS6-3.pdf)        | [Dataset](http://www.cp.jku.at/datasets/musiclef/index.html) |
| Moodo                | The Moodo dataset: Integrating user context with emotional and color perception of music for affective music information retrieval | [Paper](https://www.tandfonline.com/doi/pdf/10.1080/09298215.2017.1333518?casa_token=GxB97r7M-WMAAAAA:7ZfS-mY7f3WTP0FBbpiaSIdU-tcRXdIIwCiLLCG8ghkw_FTRn_Ha3cPD7s_6i29RwLBd6EPJmg) | [Dataset](http://moodo.musiclab.si/)                         |
| ALF-200k             | ALF-200k: Towards Extensive Multimodal Analyses of Music Tracks and Playlists | [Paper](https://dbis-informatik.uibk.ac.at/sites/default/files/2018-04/ecir-2018-alf.pdf) | [Dataset](https://github.com/dbis-uibk/ALF200k)              |
| MQA                  | Can Image Captioning Help Passage Retrieval in Multimodal Question Answering? | [Paper](https://www.springerprofessional.de/en/can-image-captioning-help-passage-retrieval-in-multimodal-questi/16626696) | [Dataset](https://huggingface.co/datasets/clips/mqa)         |
| WAT2019              | WAT2019: English-Hindi Translation on Hindi Visual Genome Dataset | [Paper](https://github.com/sheffieldnlp/mlt)                 | [Dataset](https://aclanthology.org/L18-1602.pdf)             |
| ViTT                 | Multimodal Pretraining for Dense Video Captioning            | [Paper](https://arxiv.org/pdf/2011.11760.pdf)                | [Dataset](https://github.com/google-research-datasets/Video-Timeline-Tags-ViTT) |
| MTD                  | MTD: A Multimodal Dataset of Musical Themes for MIR Research | [Paper](https://transactions.ismir.net/articles/10.5334/tismir.68/) | [Dataset](https://www.audiolabs-erlangen.de/resources/MIR/MTD) |
| MusiClef             | A professionally annotated and enriched multimodal data set on popular music | [Paper](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.302.2718&rep=rep1&type=pdf) | [Dataset](http://www.cp.jku.at/datasets/musiclef/index.html) |
| Schubert Winterreise | Schubert Winterreise dataset: A multimodal scenario for music analysis | [Paper](https://dl.acm.org/doi/pdf/10.1145/3429743)          | [Dataset](https://zenodo.org/record/3968389#.YcQrk2hBxPY)    |
| WIT                  | WIT: Wikipedia-based Image Text Dataset for Multimodal Multilingual Machine Learning | [Paper](https://arxiv.org/pdf/2103.01913.pdf)                | [Dataset](https://github.com/google-research-datasets/wit)   |

1. **Question Answering**

| **Dataset**        | **Title of the Paper**                                       | **Link of the Paper**                                        | **Link of the Dataset**                                      |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| MQA                | A Dataset for Multimodal Question Answering in the Cultural Heritage Domain | [Paper](https://aclanthology.org/W16-4003.pdf)               | -                                                            |
| MovieQA            | Movieqa: Understanding stories in movies through question-answering MovieQA | [Paper](https://arxiv.org/pdf/1512.02902.pdf)                | [Dataset](https://github.com/makarandtapaswi/MovieQA_CVPR2016) |
| PororoQA           | Deep story video story qa by deep embedded memory networks   | [Paper](https://arxiv.org/ftp/arxiv/papers/1707/1707.00836.pdf) | [Dataset](https://github.com/Kyung-Min/Deep-Embedded-Memory-Networks) |
| MemexQA            | MemexQA: Visual Memex Question Answering                     | [Paper](https://arxiv.org/pdf/1708.01336.pdf)                | [Dataset](https://memexqa.cs.cmu.edu/)                       |
| VQA                | Making the V in VQA matter: Elevating the role of image understanding in Visual Question Answering | [Paper](https://arxiv.org/pdf/1612.00837.pdf)                | [Dataset](https://visualqa.org/)                             |
| TDIUC              | An analysis of visual question answering algorithms          | [Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Kafle_An_Analysis_of_ICCV_2017_paper.pdf) | [Dataset](https://kushalkafle.com/projects/tdiuc.html)       |
| TGIF-QA            | TGIF-QA: Toward spatio-temporal reasoning in visual question answering | [Paper](https://arxiv.org/pdf/1704.04497.pdf)                | [Dataset](https://github.com/YunseokJANG/tgif-qa)            |
| MSVD QA, MSRVTT QA | Video question answering via attribute augmented attention network learning | [Paper](https://arxiv.org/pdf/1707.06355.pdf)                | [Dataset](https://github.com/xudejing/video-question-answering) |
| YouTube2Text       | Video Question Answering via Gradually Refined Attention over Appearance and Motion | [Paper](http://staff.ustc.edu.cn/~hexn/papers/mm17-videoQA.pdf) | [Dataset](https://github.com/topics/youtube2text)            |
| MovieFIB           | A dataset and exploration of models for understanding video data through fill-in-the-blank question-answering | [Paper](https://arxiv.org/pdf/1611.07810.pdf)                | [Dataset](https://github.com/teganmaharaj/MovieFIB/blob/master/README.md) |
| Video Context QA   | Uncovering the temporal context for video question answering | [Paper](https://arxiv.org/pdf/1511.04670.pdf)                | [Dataset](https://github.com/ffmpbgrnn/VideoQA)              |
| MarioQA            | Marioqa: Answering questions by watching gameplay videos     | [Paper](https://arxiv.org/pdf/1612.01669.pdf)                | [Dataset](https://github.com/JonghwanMun/MarioQA)            |
| TVQA               | Tvqa: Localized, compositional video question answering      | [Paper](https://arxiv.org/pdf/1809.01696.pdf)                | [Dataset](https://tvqa.cs.unc.edu/)                          |
| VQA-CP v2          | Don’t just assume; look and answer: Overcoming priors for visual question answering | [Paper](https://arxiv.org/pdf/1712.00377.pdf)                | [Dataset](https://github.com/cdancette/vqa-cp-leaderboard)   |
| RecipeQA           | RecipeQA: A Challenge Dataset for Multimodal Comprehension of Cooking Recipes | [Paper](https://arxiv.org/pdf/1809.00812.pdf)                | [Dataset](https://hucvl.github.io/recipeqa/)                 |
| GQA                | GQA: A new dataset for real-world visual reasoning and compositional question answering | [Paper](https://arxiv.org/pdf/1902.09506v3.pdf)              | [Dataset](https://github.com/leaderj1001/Vision-Language)    |
| Social IQ          | Social-iq: A question answering benchmark for artificial social intelligence | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zadeh_Social-IQ_A_Question_Answering_Benchmark_for_Artificial_Social_Intelligence_CVPR_2019_paper.pdf) | [Dataset](https://github.com/A2Zadeh/CMU-MultimodalSDK)      |
| MIMOQA             | MIMOQA: Multimodal Input Multimodal Output Question Answering | [Paper](https://aclanthology.org/2021.naacl-mai)             | -                                                            |

1. **Summarization**

| **Dataset**                       | **Title of the Paper**                                       | **Link of the Paper**                                        | **Link of the Dataset**                                      |
| --------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| SumMe                             | Tvsum: Summarizing web videos using titles                   | [Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Song_TVSum_Summarizing_Web_2015_CVPR_paper.pdf) | [Dataset](https://github.com/yalesong/tvsum)                 |
| TVSum                             | Creating summaries from user videos                          | [Paper](https://gyglim.github.io/me/papers/GygliECCV14_vsum.pdf) | [Dataset](https://gyglim.github.io/me/vsum/index.html)       |
| QFVS                              | Query-focused video summarization: Dataset, evaluation, and a memory network based approach | [Paper](https://arxiv.org/abs/1707.04960)                    | [Dataset](https://www.aidean-sharghi.com/cvpr2017)           |
| MMSS                              | Multi-modal Sentence Summarization with Modality Attention and Image Filtering | [Paper](https://www.ijcai.org/proceedings/2018/0577.pdf)     | -                                                            |
| MSMO                              | MSMO: Multimodal Summarization with Multimodal Output        | [Paper](https://aclanthology.org/D18-1448.pdf)               | -                                                            |
| Screen2Words                      | Screen2Words: Automatic Mobile UI Summarization with Multimodal Learning | [Paper](https://arxiv.org/pdf/2108.03353.pdf)                | [Dataset](https://github.com/google-research-datasets/screen2words) |
| AVIATE                            | IEMOCAP: interactive emotional dyadic motion capture database | [Paper](https://link.springer.com/content/pdf/10.1007/s10579-008-9076-6.pdf) | [Dataset](https://sail.usc.edu/software/databases/)          |
| Multimodal Microblog Summarizaion | On Multimodal Microblog Summarization                        | [Paper](https://ieeexplore.ieee.org/document/9585070)        | -                                                            |

1. **Human Computer Interaction**

| **Dataset**              | **Title of the Paper**                                       | **Link of the Paper**                                        | **Link of the Dataset**                                      |
| ------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| CAUVE                    | CUAVE: A new audio-visual database for multimodal human-computer interface research | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5745028) | [Dataset](http://people.csail.mit.edu/siracusa/avdata/)      |
| MHAD                     | Berkeley mhad: A comprehensive multimodal human action database | [Paper](https://ieeexplore.ieee.org/document/6474999)        | [Dataset](https://tele-immersion.citris-uc.org/berkeley_mhad) |
| Multi-party interactions | A Multi-party Multi-modal Dataset for Focus of Visual Attention in Human-human and Human-robot Interaction | [Paper](https://aclanthology.org/L16-1703.pdf)               | -                                                            |
| MHHRI                    | Multimodal human-human-robot interactions (mhhri) dataset for studying personality and engagement | [Paper](https://ieeexplore.ieee.org/document/8003432)        | [Dataset](https://www.cl.cam.ac.uk/research/rainbow/projects/mhhri/) |
| Red Hen Lab              | Red Hen Lab: Dataset and Tools for Multimodal Human Communication Research | [Paper](https://link.springer.com/content/pdf/10.1007/s13218-017-0505-9.pdf) | -                                                            |
| EMRE                     | Generating a Novel Dataset of Multimodal Referring Expressions | [Paper](https://aclanthology.org/W19-0507.pdf)               | [Dataset](https://github.com/VoxML/public-data/tree/master/EMRE/HIT) |
| Chinese Whispers         | Chinese whispers: A multimodal dataset for embodied language grounding | [Paper](https://www.researchgate.net/publication/341294259_Chinese_Whispers_A_Multimodal_Dataset_for_Embodied_Language_Grounding) | [Dataset](https://zenodo.org/record/4587308#.YbJEctBBxPZ)    |
| uulmMAC                  | The uulmMAC database—A multimodal affective corpus for affective computing in human-computer interaction | [Paper](https://www.mdpi.com/1424-8220/20/8/2308)            | [Dataset](https://neuro.informatik.uni-ulm.de/TC9/tools-and-data-sets/uulmmac-database/) |

1. **Semantic Analysis**

| **Dataset**                                    | **Title of the Paper**                                       | **Link of the Paper**                                        | **Link of the Dataset**                                      |
| ---------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| WN9-IMG                                        | Image-embodied Knowledge Representation Learning             | [Paper](https://www.ijcai.org/proceedings/2017/0438.pdf)     | [Dataset](https://github.com/xrb92/IKRL)                     |
| Wikimedia Commons                              | A Dataset and Reranking Method for Multimodal MT of User-Generated Image Captions | [Paper](https://aclanthology.org/W18-1814.pdf)               | [Dataset](https://commons.wikimedia.org/wiki/Main_Page)      |
| Starsem18-multimodalKB                         | A Multimodal Translation-Based Approach for Knowledge Graph Representation Learning | [Paper](https://aclanthology.org/S18-2027.pdf)               | [Dataset](https://github.com/UKPLab/starsem18-multimodalKB)  |
| MUStARD                                        | Towards Multimodal Sarcasm Detection                         | [Paper](https://arxiv.org/pdf/1906.01815.pdf)                | [Dataset](https://github.com/soujanyaporia/MUStARD)          |
| YouMakeup                                      | YouMakeup: A Large-Scale Domain-Specific Multimodal Dataset for Fine-Grained Semantic Comprehension | [Paper](https://aclanthology.org/D19-1517.pdf)               | [Dataset](https://github.com/AIM3-RUC/YouMakeup)             |
| MDID                                           | Integrating Text and Image: Determining Multimodal Document Intent in Instagram Posts | [Paper](https://arxiv.org/pdf/1904.09073.pdf)                | [Dataset](https://github.com/karansikka1/documentIntent_emnlp19) |
| Social media posts from Flickr (Mental Health) | Inferring Social Media Users’ Mental Health Status from Multimodal Information | [Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7147779/) | [Dataset](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7147779/) |
| Twitter MEL                                    | Building a Multimodal Entity Linking Dataset From Tweets Building a Multimodal Entity Linking Dataset From Tweets | [Paper](https://github.com/drmuskangarg/Multimodal-datasets/blob/main/aclanthology.org) | [Dataset](https://github.com/OA256864/MEL_Tweets)            |
| MultiMET                                       | MultiMET: A Multimodal Dataset for Metaphor Understanding    | [Paper](https://aclanthology.org/2021.acl-long.249.pdf)      | -                                                            |
| MSDS                                           | Multimodal Sarcasm Detection in Spanish: a Dataset and a Baseline | [Paper](https://arxiv.org/pdf/2105.05542.pdf)                | [Dataset](https://zenodo.org/record/4701383)                 |

1. **Miscellaneous**

| **Dataset**      | **Title of the Paper**                                       | **Link of the Paper**                                        | **Link of the Dataset**                                      |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| MS COCO          | Microsoft COCO: Common objects in context                    | [Paper](https://link.springer.com/chapter/10.1007/978-3-319-10602-1_48) | [Dataset](https://github.com/topics/mscoco-dataset)          |
| ILSVRC           | ImageNet Large Scale Visual Recognition Challenge            | [Paper](https://arxiv.org/pdf/1409.0575.pdf)                 | [Dataset](https://image-net.org/download.php)                |
| YFCC100M         | YFCC100M: The new data in multimedia research                | [Paper](https://arxiv.org/pdf/1503.01817.pdf)                | [Dataset](https://github.com/chi0tzp/YFCC100M-Downloader)    |
| COGNIMUSE        | COGNIMUSE: a multimodal video database annotated with saliency, events, semantics and emotion with application to summarization | [Paper](https://jivp-eurasipjournals.springeropen.com/track/pdf/10.1186/s13640-017-0194-1.pdf) | [Dataset](https://cognimuse.cs.ntua.gr/research_datasets)    |
| SNAG             | SNAG: Spoken Narratives and Gaze Dataset                     | [Paper](https://aclanthology.org/P18-2022.pdf)               | [Dataset](https://mvrl-clasp.github.io/SNAG/)                |
| UR-Funny         | UR-FUNNY: A Multimodal Language Dataset for Understanding Humor | [Paper](https://arxiv.org/pdf/1904.06618.pdf)                | [Dataset](https://github.com/ROC-HCI/UR-FUNNY/blob/master/UR-FUNNY-V1.md) |
| Bag-of-Lies      | Bag-of-Lies: A Multimodal Dataset for Deception Detection    | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9025340) | [Dataset](http://iab-rubric.org/resources/BagLies.html)      |
| MARC             | A Recipe for Creating Multimodal Aligned Datasets for Sequential Tasks | [Paper](https://aclanthology.org/2020.acl-main.440.pdf)      | [Dataset](https://github.com/microsoft/multimodal-aligned-recipe-corpus) |
| MuSE             | MuSE: a Multimodal Dataset of Stressed Emotion               | [Paper](https://aclanthology.org/2020.lrec-1.187.pdf)        | [Dataset](http://lit.eecs.umich.edu/downloads.html)          |
| BabelPic         | Fatality Killed the Cat or: BabelPic, a Multimodal Dataset for Non-Concrete Concept | [Paper](https://aclanthology.org/2020.acl-main.425.pdf)      | [Dataset](https://sapienzanlp.github.io/babelpic/)           |
| Eye4Ref          | Eye4Ref: A Multimodal Eye Movement Dataset of Referentially Complex Situations | [Paper](https://aclanthology.org/2020.lrec-1.292.pdf)        | -                                                            |
| Troll Memes      | A Dataset for Troll Classification of TamilMemes             | [Paper](https://aclanthology.org/2020.wildre-1.2.pdf)        | [Dataset](https://github.com/sharduls007/TamilMemes)         |
| SEMD             | EmoSen: Generating sentiment and emotion controlled responses in a multimodal dialogue system | [Paper](https://www.computer.org/csdl/journal/ta/5555/01/09165162/1mcQTrYsXbG) | -                                                            |
| Chat talk Corpus | Construction and Analysis of a Multimodal Chat-talk Corpus for Dialog Systems Considering Interpersonal Closeness | [Paper](https://aclanthology.org/2020.lrec-1.56.pdf)         | -                                                            |
| EMOTyDA          | Towards Emotion-aided Multi-modal Dialogue Act Classification | [Paper](https://aclanthology.org/2020.acl-main.402.pdf)      | [Dataset](https://github.com/sahatulika15/EMOTyDA)           |
| MELINDA          | MELINDA: A Multimodal Dataset for Biomedical Experiment Method Classification | [Paper](https://arxiv.org/pdf/2012.09216.pdf)                | [Dataset](https://github.com/PlusLabNLP/melinda)             |
| NewsCLIPpings    | NewsCLIPpings: Automatic Generation of Out-of-Context Multimodal Media | [Paper](https://aclanthology.org/2021.emnlp-main.545.pdf)    | [Dataset](https://github.com/g-luo/news_clippings)           |
| R2VQ             | Designing Multimodal Datasets for NLP Challenges             | [Paper](https://arxiv.org/pdf/2105.05999.pdf)                | [Dataset](https://competitions.codalab.org/competitions/34056) |
| M2H2             | M2H2: A Multimodal Multiparty Hindi Dataset For Humor Recognition in Conversations |                                                              |                                                              |
