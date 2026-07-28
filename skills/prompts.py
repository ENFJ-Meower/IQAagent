DISTORTION_DETECTION_PROMPT = """You are an expert in image quality assessment.
Examine this image carefully and identify all visible quality distortions.

Only select from these distortion types:
["Blurs", "Noise", "Compression", "Brightness change", "Color distortions", "Sharpness", "Contrast"]

Instructions:
- Select only distortions that are clearly visible
- If the image looks clean and high quality, return an empty list
- Do not mention dataset names, image IDs, or file names

Return ONLY valid JSON with no extra text:
{"distortion_set": {"Global": ["<distortion_1>", "<distortion_2>"]}}"""


DISTORTION_ANALYSIS_PROMPT = """You are an expert in image distortion analysis.
Assess the severity of each detected distortion in this image.

Detected distortions: {distortion_list}

Severity scale:
- none: not visible
- slight: barely noticeable, does not affect viewing experience
- moderate: noticeable, somewhat affects viewing experience
- severe: clearly visible, significantly degrades quality
- extreme: dominates the image, content barely discernible

Return ONLY valid JSON with no extra text:
{{
  "distortion_analysis": [
    {{
      "type": "<distortion_type>",
      "severity": "<none|slight|moderate|severe|extreme>",
      "explanation": "<one sentence of visual evidence>"
    }}
  ]
}}"""


UNIFIED_PROMPT = """You are a technical image quality assessor. Complete THREE tasks in ONE response.

RULE: Evaluate PIXEL-LEVEL technical quality ONLY. Ignore content, composition, artistic style, emotions.
ONLY judge: sharpness/blur, noise/grain, compression artifacts, exposure errors, color accuracy.

=== TASK 1: Distortion Detection ===
Identify all visible distortions. Only use types from:
["Blurs", "Noise", "Compression", "Brightness change", "Color distortions", "Sharpness", "Contrast"]

=== TASK 2: Distortion Analysis ===
For each detected distortion, assess severity: none / slight / moderate / severe / extreme

=== TASK 3: Quality Scoring ===
Objective tool measurements (calibrated to MOS 1-5 scale):
- Global sharpness MOS: {sharpness_mos:.2f}   (1=very blurry, 5=very sharp)
- BRISQUE MOS:          {brisque_mos:.2f}      (1=poor signal statistics, 5=clean natural image)
- Noise raw:            {noise:.3f}            (higher = more grain/noise)
- Composite tool MOS:   {tool_composite:.2f}   (weighted average — this is your baseline)
- Worst patch sharpness MOS: {local_sharp_mos:.2f}  (sharpness of the blurriest corner region)
- Worst patch noise raw:     {local_noise:.3f}       (noise level of the noisiest corner region)

Visual evidence:
- Image 1: Original photo
- Image 2: Noise residual map — bright = noise/grain; near-black = clean
- Image 3: Edge gradient map — bright crisp edges = sharp; diffuse/dim = blurry

MANDATORY SCORING RULES:
1. Start from composite tool MOS = {tool_composite:.2f} as baseline
2. If worst-patch sharpness MOS ({local_sharp_mos:.2f}) is more than 0.5 below global ({sharpness_mos:.2f}), adjust down 0.2-0.4
3. Adjust at most ±0.8 total from baseline
4. "No distortions detected" does NOT mean perfect quality — tool scores are your anchor
5. Score 4.5-5.0 REQUIRES: crisp gradient edges AND near-black noise map AND BRISQUE MOS ≥ 4.0
6. Output PRECISE DECIMAL (e.g., 2.8, 3.3, 3.7, 4.2) — avoid integers or .0/.5 endings

Quality reference:
1.0-1.9 = Severe degradation  2.0-2.9 = Noticeable issues  3.0-3.9 = Acceptable
4.0-4.4 = Good quality        4.5-5.0 = Exceptional (rare)

Return ONLY valid JSON with no extra text:
{{
  "distortion_set": {{"Global": ["<type1>", "<type2>"]}},
  "distortion_analysis": [
    {{"type": "<type>", "severity": "<level>", "explanation": "<one sentence>"}}
  ],
  "quality_score": <precise float 1.0-5.0>,
  "primary_distortion": "<main issue or 'none'>",
  "reasoning": "<2-3 sentences citing tool scores and map observations>"
}}"""


SCORING_PROMPT = """You are a technical image quality assessor. Evaluate PIXEL-LEVEL technical quality ONLY.

RULE: Do NOT consider content, composition, subject matter, artistic style, or emotions. Ignore whether the scene is beautiful or interesting.
ONLY judge: sharpness/blur, noise/grain, compression artifacts, exposure errors, color accuracy.
A stunning landscape with soft focus is LOW quality. A plain wall in perfect focus is HIGH quality.

Objective tool measurements (calibrated to MOS 1-5 scale):
- Global sharpness MOS: {sharpness_mos:.2f}   (1=very blurry, 5=very sharp)
- BRISQUE MOS:          {brisque_mos:.2f}      (1=poor signal statistics, 5=clean natural image)
- Noise raw:            {noise:.3f}            (higher = more grain/noise)
- Composite tool MOS:   {tool_composite:.2f}   (weighted average — this is your baseline)
- Worst patch sharpness MOS: {local_sharp_mos:.2f}  (sharpness of the blurriest corner region)
- Worst patch noise raw:     {local_noise:.3f}       (noise level of the noisiest corner region)

Distortion analysis from detection step:
{distortion_summary}

Visual evidence provided:
- Image 1: Original photo
- Image 2: Noise residual map — bright = noise/grain present; near-black = clean
- Image 3: Edge gradient map — bright crisp edges = sharp; diffuse/dim = blurry or soft

MANDATORY SCORING RULES:
1. Start from composite tool MOS = {tool_composite:.2f} as your baseline score
2. If worst-patch sharpness MOS ({local_sharp_mos:.2f}) is more than 0.5 below global sharpness MOS ({sharpness_mos:.2f}), this indicates localized blur — adjust baseline down by 0.2-0.4
3. Adjust by at most ±0.8 total only if visual maps give clear contradicting evidence
4. "No distortions detected" means no dominant distortions — NOT perfect quality. Tool scores remain your anchor.
5. Score 4.5-5.0 REQUIRES: gradient map shows crisp bright edges AND noise map is near-black AND BRISQUE MOS ≥ 4.0
6. Output a PRECISE DECIMAL reflecting tool measurements (e.g., 2.8, 3.3, 3.7, 4.2) — avoid clustering at integers or .0/.5

Quality reference:
1.0-1.9 = Severe technical degradation, distracting to any viewer
2.0-2.9 = Noticeable quality issues, below-average for consumer photography
3.0-3.9 = Acceptable quality, typical web/consumer photo with minor imperfections
4.0-4.4 = Good quality, only very subtle technical issues
4.5-5.0 = Exceptional — near-perfect sharpness, minimal noise, accurate exposure (rare)

Step-by-step reasoning:
1. Composite tool score is {tool_composite:.2f} → baseline
2. Worst patch sharpness MOS is {local_sharp_mos:.2f} vs global {sharpness_mos:.2f} → local quality delta?
3. Noise residual map observation: [describe brightness pattern]
4. Gradient map observation: [describe edge strength and clarity]
5. Final adjustment from baseline: [justify, must stay within ±0.8 of {tool_composite:.2f}]

Return ONLY valid JSON with no extra text:
{{"quality_score": <precise float 1.0-5.0>, "primary_distortion": "<main technical issue or 'none'>", "reasoning": "<2-3 sentences citing composite tool score, patch quality, and map observations>"}}"""


COMPARATOR_PROMPT = """You are a perceptual image quality expert.
Compare the overall quality of these two images.

Image A is the query image (left/first).
Image B is a reference image with known quality level: {ref_description}

Consider: sharpness, noise, compression artifacts, exposure, color accuracy, and overall visual appeal.

Which image has better overall perceptual quality?
Answer with a single letter only: A or B"""
