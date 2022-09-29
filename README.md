<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->




<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="Talk2DB/static/img/logo.bmp" alt="Logo" width="180" height="80">
  </a>

  <h3 align="center">TEMPSQL</h3>

  <p align="center">
    TEMPLATE BASED TEXT-TO-SQL GENERATION ON SINGLE SOURCE DATABASE
    <br />
    <a href="https://github.com/AbhishekBhattGitHub/TempSQL/blob/main/Reports/TEMPSQL_%20Report.pdf"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·
   <!-- <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a> -->
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#the-problem-statement">Problem Statement</a></li>
        <li><a href="#our-approach">Our Approach</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#prerequisites">Prerequisites</a></li>
    <li><a href="#contact">Contact</a></li>
   
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com)  -->



### The Problem Statement
As the industry reserve of data kept increasing in the past two decades, data storage formats keep evolving from structured to un-structured data. But still the majority of data is kept and managed in Structured Tabular Format in relational database and queried through SQL. 
As SQL largely remained a developer or DBA language since its inception, the non-technical users or top-tier management of any company has to rely on the business analytics software like Tableau, Power BI or their inhouse software to gain insights from their data repository.
These analytical tools succeeded in abstracting the SQL complexity from the users by hard coding the queries based on input provided by the users. However, they do not support on-demand natural language querying which has been a highlight feature for the querying systems of unstructured data such as the web search engines in the past couple of decades.
A lot of advancements in query understanding and information retrieval techniques has been made possible through AI/ML research specifically in NLP domain. But the research results of Text-to-SQL conversion are yet to be successfully commercialized because state-of-the art neural models like RAT-SQL [2], RYAL-SQL [3] etc. on the SPIDER dataset yields only 60–65 exact matching accuracy.

### Our Approach
In this project, we have attempted to develop an end-to-end commercial solution for Text-to-SQL problem which can be easily integrated with existing systems in the industry. Our proposed solution consists of two modules:

a)	**TempSQL**: An attention based Bidirectional Seq2Seq Model trained on synthetic data generated from templates

b)	**Talk2DB**: An interactive interface for inference where user can leverage autocomplete feature based on Language Modelling on Questions dataset and get desired results

Both of the above modules will be discussed in details in coming sections of the report. We have tested our solution on two diverse multi-table databases of SPIDER dataset viz. college_2 and store_1 and got the test accuracy of 72% and 81% respectively.


### Built With

The major frameworks that we have used in this project.
* [Tensorflow](https://www.tensorflow.org)
* [NLTK](https://www.nltk.org)
* [Flask](https://flask.palletsprojects.com/en/2.0.x/)
* [JQuery](https://jquery.com)




<!-- GETTING STARTED -->
## Getting Started

Python Version Used: 3.8

Clone the project from github and follow the below steps:
### Prerequisites

Install the following python libraries using pip
*  numpy
  ```sh
  pip install numpy
  ```
*  nltk
*  flask
*  os
*  pandas
*  PyDictionary
*  re
*  pandas
*  sqlite3
*  pathlib
*  collections
*  tensorflow 
*  matplotlib
*  sklearn
*  unicodedata
*  io
*  time

### Dataset

* [SPIDER] (https://yale-lily.github.io/spider) <br>
The processed augmented data has been placed [here](https://github.com/AbhishekBhattGitHub/TempSQL/tree/main/DataAugmentation) 


<!-- CONTACT -->
## Contact

Abhishek Bhatt - [@abhishek_bhatt](https://www.linkedin.com/in/abhishek-bhatt-03902019/) - to.abhishek.bhatt@gmail.com <br>

Anand Rai - [@anand_jpc](https://twitter.com/anand_jpc) - raianand.1991@gmail.com










<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
