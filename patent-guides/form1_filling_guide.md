# 📋 Form 1 Field-by-Field Guide + Document Preparation Checklist

> [!NOTE]
> **Your situation:** College student → Individual applicant → Natural Person → Filing provisional specification for a Neuro-Symbolic RAG system → Online e-filing at ipindiaonline.gov.in

---

## Part A: Documents to Prepare (In Order)

Prepare these **before** you start the e-filing process:

### ✅ Document Checklist

| # | Document | Priority | Status | Notes |
|---|---|---|---|---|
| 1 | **Form 2 — Provisional Specification** (PDF) | 🔴 Must | ✅ Ready | Convert your `PROVISIONAL_PATENT_SPECIFICATION.md` to PDF |
| 2 | **Form 1 — Application for Grant of Patent** | 🔴 Must | ❌ Fill online | Filled on e-filing portal (guide below) |
| 3 | **Form 28 — Declaration of Natural Person** | 🔴 Must | ❌ Fill online | Required for **80% fee discount** |
| 4 | **Abstract** (150 words) | 🟡 Recommended | ❌ Prepare | Short summary of your invention |
| 5 | **Drawings** (PDF) | 🟡 Recommended | ✅ Ready | Your architecture diagrams from the spec |
| 6 | **Aadhaar Card** (scan) | 🔴 Must | — | For portal registration & identity |
| 7 | **Form 3 — Foreign Filing Statement** | ⚪ Only if needed | Skip | Only if you've filed the same patent in another country |
| 8 | **Form 26 — Patent Agent Authorization** | ⚪ Only if needed | Skip | Only if using a patent agent |
| 9 | **Form 5 — Declaration of Inventorship** | 🟡 Recommended | ❌ Fill online | If applicant ≠ inventor (skip if you are the sole inventor-applicant) |

---

### 📄 Preparing Form 2 (Your Provisional Specification as PDF)

Your `PROVISIONAL_PATENT_SPECIFICATION.md` is already well-drafted. To submit it:

1. Open the file in VS Code or any Markdown viewer
2. **Export/Print to PDF** (Ctrl+Shift+P → "Markdown: Export PDF" or use browser print)
3. Ensure the PDF is **under 10MB** (should be well under)
4. Make sure all diagrams/tables render properly in the PDF
5. **Do NOT include claims** — provisional specs don't need them (yours correctly doesn't)

---

### 📝 Preparing the Abstract (Write This)

Write a **150-word summary** of your invention. Here's a ready-to-use draft:

> **Abstract**
>
> A computer-implemented Neuro-Symbolic Agentic system and method for Retrieval-Augmented Generation from unstructured PDF documents, employing a novel Tri-Layer Cognitive Architecture. The first layer implements an agentic control loop with autonomous query routing, hallucination grading, and adaptive retry mechanisms for self-correcting question-answering. The second layer maintains a hybrid knowledge store combining a vector embedding database with a symbolic knowledge graph employing Leiden community detection, enabling both semantic similarity search and structured entity-relation traversal. The third layer implements a Graph-Augmented Fusion Retrieval algorithm using Weighted Reciprocal Rank Fusion to combine vector cosine similarity scores with knowledge graph traversal scores. The system features structure-preserving PDF extraction rendering documents as Markdown, three-tier boundary-aware semantic chunking, and resource-aware model orchestration enabling deployment on consumer hardware with as little as 8GB RAM without cloud API dependencies.

Save this as a separate PDF or include it at the end of your Form 2 PDF.

---

## Part B: Form 1 — Field-by-Field Guide

### Section 1: Type of Application

| Field | What to Select | Explanation |
|---|---|---|
| **Application Type** | **☑ Ordinary Application** | You're filing for the first time in India, not claiming priority from another country |
| Convention Application | ☐ Leave unchecked | Only if you filed in another country first (Paris Convention) |
| PCT National Phase | ☐ Leave unchecked | Only for international PCT applications |
| Divisional Application | ☐ Leave unchecked | Only if splitting from an existing application |
| Patent of Addition | ☐ Leave unchecked | Only if improving an existing patented invention |

---

### Section 2: Specification Type

| Field | What to Select |
|---|---|
| **Specification Filed** | **☑ Provisional** |
| Complete | ☐ Leave unchecked |

---

### Section 3A: Applicant Details

> [!IMPORTANT]
> Fill this with YOUR personal details. If you have a co-inventor (e.g., project guide/professor), add them too.

| Field | What to Fill | Example |
|---|---|---|
| **Name** | Your full legal name (as on Aadhaar) | `Shikhar Purwar` |
| **Address** | Full postal address with PIN code | `123, ABC Colony, City, State - 123456, India` |
| **Nationality** | `Indian` | |
| **Country of Residence** | `India` | |

**If you have a co-applicant (professor/college):**
- Click "Add Applicant" and fill their details too
- You'll need their consent/signature

---

### Section 3B: Category of Applicant

| Field | What to Select | Why |
|---|---|---|
| **Category** | **☑ Natural Person** | This gives you the **cheapest fees** (₹1,600 for provisional) |
| Startup | ☐ | Only if you have DPIIT-recognized startup |
| Small Entity | ☐ | Only for registered small enterprises |
| Others (Large Entity) | ☐ | For corporations — most expensive |

> [!TIP]
> **You MUST also file Form 28** to claim this reduced fee. Without Form 28, you'll be charged the "Others" (highest) rate!

---

### Section 4: Inventor Details

| Field | What to Fill | Example |
|---|---|---|
| **Name of Inventor** | Your full name | `Shikhar Purwar` |
| **Address** | Same as applicant address | `123, ABC Colony, City, State - 123456` |
| **Nationality** | `Indian` | |
| **Is the applicant the inventor?** | **☑ Yes** | Check this if YOU are the inventor |

**If there are co-inventors:**
- Click "Add Inventor" for each additional person
- Add their name, address, nationality
- This includes your project guide if they contributed to the invention

---

### Section 5: Title of the Invention

| Field | What to Fill |
|---|---|
| **Title** | `Neuro-Symbolic Agentic System and Method for Retrieval-Augmented Generation from Unstructured Documents Using Tri-Layer Cognitive Architecture` |

> [!NOTE]
> This is the same title as in your Form 2 (Provisional Specification). **They must match exactly.**

---

### Section 6: Prior Application Details (Convention/PCT Priority)

| Field | What to Fill |
|---|---|
| **Have you filed this application in any other country?** | **No** |
| Country | — Leave blank |
| Application Number | — Leave blank |
| Filing Date | — Leave blank |

Skip this entire section — it only applies if you've filed abroad first.

---

### Section 7: Address for Service in India

| Field | What to Fill | Example |
|---|---|---|
| **Address for Correspondence** | Your address where you want to receive official letters from the Patent Office | Same as your applicant address |
| **Email** | Your email (very important — most communication is via email now) | `your.email@gmail.com` |
| **Phone** | Your mobile number | `+91-XXXXXXXXXX` |

> [!IMPORTANT]
> Use an email and phone you **check regularly**. Missing a Patent Office communication can kill your application.

---

### Section 8: Declarations

Check the applicable declarations:

| Declaration | Check? | Meaning |
|---|---|---|
| **"I/We, the applicant(s) am/are the true and first inventor(s)"** | **☑ Yes** | You invented this yourself |
| "The invention has been communicated from outside India" | ☐ No | Your invention was developed in India |
| "The application has been filed in other countries" | ☐ No | First-time filing |
| "I/We undertake to file Form 3 within 6 months" | ☐ No | Not applicable since no foreign filing |

---

### Section 9: Fee Payment

| Field | What to Fill |
|---|---|
| **Fee Category** | Natural Person |
| **Amount** | ₹1,600 (for provisional, electronic filing, up to 30 pages) |
| **Payment Mode** | Online — UPI / Net Banking / Debit Card / Credit Card |

**Extra page fee:** If your Form 2 (specification) exceeds **30 pages**, add ₹160 per extra page (Natural Person rate).

Your spec is ~380 lines in Markdown, which should be approximately 10–15 pages in PDF format — well within the 30-page limit.

---

### Section 10: Signature

| Field | What to Do |
|---|---|
| **Signed by** | Your full name |
| **Date** | Date of filing |
| **Digital Signature** | Not mandatory for provisional — you can upload a scanned wet signature |
| **Capacity** | `Applicant` (or `Inventor and Applicant`) |

---

## Part C: Form 28 — Declaration for Natural Person Fee

This is a simple one-page form. Fill as follows:

| Field | What to Fill |
|---|---|
| **Application Number** | Leave blank initially (you get this after Form 1 submission) — or fill after getting the number |
| **Name of Applicant** | Your full legal name |
| **Category** | **☑ Natural Person** |
| **Declaration** | "I hereby declare that I am a Natural Person as defined under the Patents Rules, 2003" |
| **Signature** | Your signature |
| **Date** | Filing date |

> [!TIP]
> On the e-filing portal, Form 28 is usually filed together with Form 1. The portal will prompt you to declare your entity type — just select "Natural Person."

---

## Part D: Step-by-Step E-Filing Walkthrough

```
1. Go to https://ipindiaonline.gov.in/epatentfiling/
2. Click "New User? Register here" → Create account
3. Verify via email → Log in
4. Dashboard → Click "File New Application" → "Patent"
5. Select: Ordinary Application → Provisional Specification
6. Fill Form 1 fields (as described above)
7. Upload Form 2 (your specification PDF)
8. Upload drawings (architecture diagram PDFs)
9. Fill Form 28 (Natural Person declaration)
10. Review all details → Pay ₹1,600 online
11. Submit → Download acknowledgement with APPLICATION NUMBER
12. 🎉 DONE! Your priority date is NOW locked in!
```

---

## Part E: Common Mistakes to Avoid

| ❌ Mistake | ✅ Fix |
|---|---|
| Title in Form 1 doesn't match Form 2 | Copy-paste the exact title from your specification |
| Forgetting Form 28 | Without it, you pay ₹8,000 instead of ₹1,600 |
| Wrong applicant category | Select "Natural Person", not "Others" |
| Specification exceeds 10MB | Compress PDF or split drawings |
| Not matching inventor and applicant names | Use your name exactly as on Aadhaar |
| Not saving the acknowledgement | Download and save the PDF immediately — it has your Application Number |
| Missing the 12-month deadline | Set reminders for filing complete specification |

---

## Pre-Filing Checklist

Use this final checklist before you hit Submit:

- [ ] Provisional Specification converted to clean PDF
- [ ] Abstract written (150 words)
- [ ] Architecture diagrams exported as PDF
- [ ] Aadhaar details ready
- [ ] Email and phone verified on portal
- [ ] ₹1,600 payment method ready (UPI/card/net banking)
- [ ] Title matches exactly between Form 1 and Form 2
- [ ] Form 28 Natural Person declaration filled
- [ ] College IP policy checked — no conflicts
- [ ] All inventor names and addresses ready
