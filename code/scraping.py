#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 11:01:21 2025

@author: konearounaromeo
"""

"""
Script unifié de web scraping Ulule - Toutes catégories
Extraction complète via pagination par offsets
"""
import time  # gestion des pauses
import pandas as pd  # manipulation et stockage des données
import os  # interaction avec le système de fichiers
from selenium import webdriver  # contrôle du navigateur
from selenium.webdriver.common.by import By  # localisation des éléments
from selenium.webdriver.support.ui import WebDriverWait  # attentes explicites
from selenium.webdriver.support import expected_conditions as EC  # conditions pour WebDriverWait
from selenium.common.exceptions import NoSuchElementException, TimeoutException  # gestion d'erreurs Selenium
import re  # expressions régulières

class UluleScraper:
    """Classe principale pour le scraping Ulule - gère le navigateur et les données"""
    
    def __init__(self, headless=True):
        self.driver = None  # instance du navigateur
        self.headless = headless  # mode headless (sans interface graphique)
        self.data = []  # liste pour stocker les données
        
    def init_driver(self):
        """Initialise le navigateur Chrome avec les options"""
        options = webdriver.ChromeOptions()  # options du navigateur
        if self.headless:
            options.add_argument("--headless")  # exécution sans interface graphique
            options.add_argument("--disable-blink-features=AutomationControlled")  # contourner la détection Selenium
            options.add_argument("--window-size=1920,1080")  # taille de la fenêtre
        self.driver = webdriver.Chrome(options=options)  # création du driver Chrome
        return self.driver  # renvoie le driver

    def close_driver(self):
        """Ferme le navigateur proprement"""
        if self.driver:
            self.driver.quit()  # ferme le navigateur
            self.driver = None  # réinitialise le driver



class ProjectLoader(UluleScraper):
    """Classe specialisee dans le chargement des projets par categories"""
    
    def get_selected_categories(self):
        """Retourne les URLs des 4 categories selectionnees avec filtre France"""
        categories = {
            "artisanat_cuisine": "https://fr.ulule.com/discover/?categories=craft-food&offset=0&statuses=all&countries=FR",
            "mode_design": "https://fr.ulule.com/discover/?categories=fashion-design&offset=0&statuses=all&countries=FR",
            "sante_bien_etre": "https://fr.ulule.com/discover/?categories=sante-bien-etre&offset=0&statuses=all&countries=FR",
            "technologie": "https://fr.ulule.com/discover/?categories=technology&offset=0&statuses=all&countries=FR"
        }
        return categories  # dictionnaire catégories -> URL
    
    def load_projects_from_category(self, url, category_name, max_clicks=10000):  
        """Charge tous les projets d'une categorie en cliquant sur 'Plus de collectes'"""
        
        if not self.driver:
            self.init_driver()  # initialise le driver si nécessaire
            
        print(f"\nCATEGORIE: {category_name}")
        print(f"Chargement: {url}")
        
        self.driver.get(url)  # ouvrir la page catégorie
        time.sleep(5)  # attendre le chargement initial

        clicks = 0  # compteur de clics
        no_new_projects_count = 0  # compteur de clics sans nouveaux projets
        previous_count = 0  # nombre de projets avant le dernier clic
        
        while clicks < max_clicks:
            try:
                projects_before = self.driver.find_elements(By.CSS_SELECTOR, "li[data-project-id]")  # projets avant clic
                count_before = len(projects_before)
                
                button = None
                selectors = [
                    "//button[.//span[contains(text(), 'Plus de collectes')]]",
                    "//button[contains(., 'Plus de collectes')]",
                    "//button[contains(., 'Charger plus')]"
                ]
                
                for selector in selectors:
                    try:
                        button = WebDriverWait(self.driver, 5).until(
                            EC.element_to_be_clickable((By.XPATH, selector))
                        )
                        if button:
                            break  # bouton trouvé
                    except:
                        continue
                
                if not button:
                    print(f"   {category_name}: Plus de bouton trouve -> fin")
                    break
                
                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", button)  # scroll
                time.sleep(1)
                self.driver.execute_script("arguments[0].click();", button)  # clic via JS
                clicks += 1
                
                time.sleep(3)  # attendre le chargement
                projects_after = self.driver.find_elements(By.CSS_SELECTOR, "li[data-project-id]")  # projets après clic
                count_after = len(projects_after)
                
                new_projects = count_after - count_before  # nouveaux projets
                
                if clicks % 50 == 0 or new_projects == 0:
                    print(f"   [{category_name}] Clic {clicks} -> {new_projects} nouveaux (Total: {count_after})")
                
                if new_projects == 0:
                    no_new_projects_count += 1
                    if no_new_projects_count >= 3:
                        print(f"   {category_name}: Aucun nouveau projet depuis 3 clics -> fin")
                        break
                else:
                    no_new_projects_count = 0
                
                if clicks > 20 and count_after == previous_count:
                    print(f"   {category_name}: Nombre stagnant -> fin")
                    break
                    
                previous_count = count_after
                
            except (TimeoutException, NoSuchElementException):
                print(f"   {category_name}: Plus de bouton -> fin")
                break
            except Exception as e:
                if "invalid session id" in str(e):
                    print(f"   {category_name}: Session expiree, redemarrage...")
                    self.close_driver()
                    self.init_driver()
                    self.driver.get(url)
                    time.sleep(5)
                    continue
                else:
                    print(f"   {category_name}: Erreur clic {clicks} -> {e}")
                    continue

        projects = self.driver.find_elements(By.CSS_SELECTOR, "li[data-project-id]")  # récupération finale
        
        urls_et_ids = []
        for p in projects:
            try:
                project_id = p.get_attribute("data-project-id")  # id projet
                link = p.find_element(By.CSS_SELECTOR, "a").get_attribute("href")  # URL projet
                urls_et_ids.append({
                    "id": project_id, 
                    "url": link,
                    "categorie_source": category_name
                })
            except Exception as e:
                continue

        print(f"   {category_name}: {len(urls_et_ids)} projets extraits apres {clicks} clics")
        
        return urls_et_ids
    
    def load_selected_categories(self, max_clicks_per_category=10000):  
        """Charge les projets des 4 categories selectionnees avec filtre France"""
        categories = self.get_selected_categories()
        all_projects = []
        
        print("DEMARRAGE DU CHARGEMENT DES 4 CATEGORIES SELECTIONNEES")
        print("MAX_CLICKS: 10000 par categorie")
        print("FILTRE: Projets France uniquement (countries=FR)")
        print("=" * 60)
        print("CATEGORIES:")
        print("- Artisanat & Cuisine")
        print("- Mode & Design")
        print("- Santé & Bien-être")
        print("- Technologie")
        print("=" * 60)
        
        for category_name, category_url in categories.items():
            try:
                category_projects = self.load_projects_from_category(
                    category_url, 
                    category_name, 
                    max_clicks=max_clicks_per_category
                )
                all_projects.extend(category_projects)
                
                print(f"RESUME {category_name.upper()}: {len(category_projects)} projets")
                
                df_temp = pd.DataFrame(all_projects)  # sauvegarde intermédiaire
                df_temp.to_csv("urls_temp.csv", index=False, encoding='utf-8')
                print(f"SAUVEGARDE TEMPORAIRE: {len(all_projects)} URLs sauvegardees")
                
                time.sleep(3)  # pause entre catégories
                
            except Exception as e:
                print(f"ERREUR categorie {category_name}: {e}")
                continue
        
        # suppression des doublons
        seen_ids = set()
        unique_projects = []
        for project in all_projects:
            if project['id'] not in seen_ids:
                seen_ids.add(project['id'])
                unique_projects.append(project)
        
        print("=" * 60)
        print(f"CHARGEMENT TERMINE: {len(unique_projects)} projets uniques au total")
        print(f"Repartition par categorie:")
        
        from collections import Counter
        categorie_counts = Counter([p['categorie_source'] for p in unique_projects])
        for categorie, count in categorie_counts.items():
            print(f"   {categorie}: {count} projets")
        
        return unique_projects  # liste finale de projets uniques


class DataExtractor(UluleScraper):
    """Classe pour extraire les differentes donnees des projets individuels"""

    def extract_titre_et_adresse(self):
        """Extrait le titre, la ville ET la categorie du projet"""
        titre, adresse, categorie = None, None, None
        
        try:
            titre_elem = self.driver.find_element(By.TAG_NAME, "h1")  # titre du projet
            titre = titre_elem.text.strip()
        except:
            pass

        try:
            elements = self.driver.find_elements(By.CSS_SELECTOR, "div.yysRhADHr.yki5sGe2.yysRhADD.yysRhADGP.yysRhADH1")  # divs contenant adresse/categorie
            
            for element in elements:
                texte = element.text.strip()
                if not texte:
                    continue
                    
                html_content = element.get_attribute('outerHTML')
                
                # Adresse (icone map-cursor)
                if 'icon-map-cursor' in html_content:
                    excluded_terms = [
                        "Artisanat", "Cuisine", "craft", "food", "cuisine", "artisanat",
                        "Mode", "Design", "Fashion", "mode", "design", "fashion",
                        "Santé", "Bien-être", "Health", "Wellness", "sante", "bien-etre",
                        "Technologie", "Technology", "tech", "techno",
                        "Bien-etre", "Sante", "Bien-être", "Santé", "Bien etre", "sante",
                        "Wellness", "Health", "Bienestar", "Salud"
                    ]
                    
                    if texte and not any(term.lower() in texte.lower() for term in excluded_terms):
                        texte_clean = re.sub(r'\(\d{2,5}\)', '', texte).strip()
                        
                        if any(char.isdigit() for char in texte_clean):
                            ville_match = re.search(r'(\d{2,5}\s+)?([A-Za-zÀ-ÿ\s\-]+)(?:\s+\d{2,5})?', texte_clean)
                            if ville_match:
                                ville = ville_match.group(2).strip()
                                if ville and len(ville) > 2:
                                    adresse = ville
                        else:
                            mots = [m for m in texte_clean.split() if len(m) > 2 and not m.isdigit()]
                            if mots:
                                adresse = texte_clean if len(texte_clean) <= 20 else mots[-1]
                
                # Categorie (icone tag)
                elif 'icon-tag' in html_content:
                    categorie = texte
            
        except Exception as e:
            print(f"Erreur extraction adresse/categorie: {e}")

        return titre, adresse, categorie  # renvoie titre, ville, catégorie

    def detecter_type_projet(self):
        """Detecte si c'est une collecte normale (euros) ou des prevenues"""
        try:
            preventes_patterns = [
                "//p[contains(., 'préventes sur')]",
                "//p[contains(., 'prévente sur')]",
                "//*[contains(., 'préventes sur')]",
                "//*[contains(., 'prévente sur')]"
            ]
            
            for pattern in preventes_patterns:
                preventes_elements = self.driver.find_elements(By.XPATH, pattern)
                if preventes_elements:
                    return "prevente"  # type prevente
            
            currency_indicators = self.driver.find_elements(By.XPATH, "//*[contains(., '€') or contains(., '$')]")
            if currency_indicators:
                return "euros"  # type euros
            
            return "inconnu"
        except:
            return "inconnu"

    def extract_montant_initial(self, type_projet):
        """Extrait le montant initial (objectif) selon le type de projet"""
        montant_initial = None

        if type_projet == "euros":
            try:
                montant_elems = self.driver.find_elements(By.CSS_SELECTOR, "span.yQP2FUQ")  # spans avec devise
                for elem in montant_elems:
                    texte = elem.text.strip()
                    if '€' in texte or '$' in texte:
                        if 'sur' in elem.find_element(By.XPATH, "..").text:
                            texte_clean = texte.replace("\u202f", "").replace("\xa0", "").replace("€", "").replace("$", "").replace(" ", "")
                            if texte_clean.isdigit():
                                montant_initial = int(texte_clean)
                                return montant_initial
                
                sur_elements = self.driver.find_elements(By.XPATH, "//*[contains(., 'sur')]")  # fallback sur "sur XXX €"
                for element in sur_elements:
                    texte = element.text.strip()
                    if 'sur' in texte.lower() and ('€' in texte or '$' in texte):
                        match = re.search(r'sur\s+([\d\s\u202f\xa0]+)\s*[€$]', texte, re.IGNORECASE)
                        if match:
                            montant_text = match.group(1).replace("\u202f", "").replace("\xa0", "").replace(" ", "")
                            if montant_text.isdigit():
                                montant_initial = int(montant_text)
                                return montant_initial
                
            except:
                pass

        elif type_projet == "prevente":
            try:
                preventes_patterns = [
                    "//p[contains(., 'préventes sur')]",
                    "//p[contains(., 'prévente sur')]"
                ]
                
                for pattern in preventes_patterns:
                    preventes_elements = self.driver.find_elements(By.XPATH, pattern)
                    for element in preventes_elements:
                        texte = element.text.strip()
                        if "sur" in texte:
                            match = re.search(r'sur\s+(\d+)', texte)
                            if match:
                                montant_initial = int(match.group(1))
                                return montant_initial
            except:
                pass

        return montant_initial

    def extract_nombre_contributions(self, type_projet):
        """Extrait le nombre de contributions/contributeurs"""
        contributions = None
        
        try:
            contributions_links = self.driver.find_elements(By.CSS_SELECTOR, "a.yysRhADEx.yC6BAXx4.yysRhADD.yysRhADDV.yysRhADE7")
            for link in contributions_links:
                if "Contributions" in link.text or "contributeurs" in link.text.lower():
                    try:
                        count_elem = link.find_element(By.CSS_SELECTOR, "div.yC6BAXx5")
                        texte = count_elem.text.strip()
                        if texte.isdigit():
                            contributions = int(texte)
                            return contributions
                    except:
                        continue
        except:
            pass
        
        try:
            contributions_divs = self.driver.find_elements(By.CSS_SELECTOR, "div.yysRhADC1, div[class*='contributions']")
            for div in contributions_divs:
                texte = div.text.strip()
                if texte.isdigit() and len(texte) <= 6:
                    contributions = int(texte)
                    return contributions
        except:
            pass
        
        try:
            elements = self.driver.find_elements(By.CSS_SELECTOR, "div[class*='yysRhAD']")
            for elem in elements:
                texte = elem.text.strip()
                if texte.isdigit() and len(texte) <= 6:
                    contributions = int(texte)
                    return contributions
        except:
            pass
        
        return contributions

    def extract_montant_collecte_actuel(self, type_projet):
        """Extrait le montant actuellement collecte selon le type de projet"""
        montant_actuel = None
        
        if type_projet == "euros":
            try:
                montant_elems = self.driver.find_elements(By.CSS_SELECTOR, "span.yQP2FUQ")
                for elem in montant_elems:
                    texte = elem.text.strip()
                    if ('€' in texte or '$' in texte) and not texte.startswith('sur'):
                        texte_clean = texte.replace("\u202f", "").replace("\xa0", "").replace("€", "").replace("$", "").replace(" ", "")
                        if texte_clean.isdigit():
                            montant_actuel = int(texte_clean)
                            return montant_actuel
                
                div_elements = self.driver.find_elements(By.CSS_SELECTOR, "div.yysRhAD7g, div[class*='amount']")
                for elem in div_elements:
                    texte = elem.text.strip()
                    if '€' in texte or '$' in texte:
                        montant_match = re.search(r'([\d\s\u202f\xa0]+)\s*[€$]', texte)
                        if montant_match:
                            montant_text = montant_match.group(1).replace("\u202f", "").replace("\xa0", "").replace(" ", "")
                            if montant_text.isdigit():
                                montant_actuel = int(montant_text)
                                return montant_actuel
                
            except Exception:
                pass
        
        elif type_projet == "prevente":
            try:
                preventes_elems = self.driver.find_elements(By.CSS_SELECTOR, "div.yysRhAD7g, div[class*='count']")
                for elem in preventes_elems:
                    texte = elem.text.strip()
                    texte_clean = texte.replace("\u202f", "").replace("\xa0", "").replace(" ", "")
                    if texte_clean.isdigit() and len(texte_clean) > 0:
                        montant_actuel = int(texte_clean)
                        return montant_actuel
            except Exception:
                pass
        
        return montant_actuel

    def extract_nombre_commentaires(self):
        """Extrait le nombre de commentaires sur le projet"""
        try:
            comment_elem = self.driver.find_element(By.XPATH, "//a[contains(@href, 'comments') or contains(., 'Commentaires')]//div[@class='yC6BAXx5']")
            texte = comment_elem.text.strip()
            if texte.isdigit():
                return int(texte)
        except:
            pass
        
        try:
            elements = self.driver.find_elements(By.XPATH, "//*[contains(., 'Commentaires')]")
            for elem in elements:
                texte = elem.text.strip()
                if "Commentaires" in texte:
                    nombres = re.findall(r'\d+', texte)
                    if nombres:
                        return int(nombres[0])
        except:
            pass
        
        return 0

    def extract_nombre_videos(self):
        """Extrait le nombre de videos dans le projet"""
        try:
            videos = self.driver.find_elements(By.XPATH, "//*[@name='play' or contains(@points, '22.667') or contains(@class, 'video') or contains(@d, 'M4.5')]")
            return len(videos)
        except:
            return 0

    def extract_nombre_images(self):
        """Extrait le nombre d'images dans le projet"""
        try:
            images_figures = self.driver.find_elements(By.CSS_SELECTOR, "figure.image, div.image, img[src*='cloudfront']")
            images_img = self.driver.find_elements(By.TAG_NAME, "img")
            all_images = list(set(images_figures + images_img))
            return len(all_images)
        except:
            return 0

    def extract_nombre_publications(self):
        """Extrait le nombre de publications/actualites du projet"""
        try:
            pub_elem = self.driver.find_element(By.XPATH, "//a[contains(@href, 'news') or contains(., 'Publications') or contains(., 'publications')]//div[@class='yC6BAXx5']")
            texte = pub_elem.text.strip()
            if texte.isdigit():
                return int(texte)
        except:
            pass
        
        try:
            elements = self.driver.find_elements(By.XPATH, "//*[contains(.,a 'Publications') or contains(., 'publications')]")
            for elem in elements:
                texte = elem.text.strip()
                if "Publications" in texte or "publications" in texte:
                    nombres = re.findall(r'\d+', texte)
                    if nombres:
                        return int(nombres[0])
        except:
            pass
        
        return 0



class UluleCompleteScraper(ProjectLoader, DataExtractor):
    """Classe principale qui combine toutes les fonctionnalités de chargement et d'extraction"""
    
    def __init__(self, headless=True):
        super().__init__(headless=headless)
        self.output_file = "ulule_4_categories_france.csv"       # fichier final
        self.partial_file = "ulule_4_categories_france_partiel.csv"  # sauvegarde intermédiaire
    
    def extraire_donnees_projet(self, item):
        """Extrait toutes les données d'un projet individuel"""
        try:
            self.driver.get(item["url"])  # ouvrir la page projet
            time.sleep(2)
            
            titre, adresse, categorie = self.extract_titre_et_adresse()  # titre, ville, categorie
            type_projet = self.detecter_type_projet()  # type projet (euros/prevente)
            montant_initial = self.extract_montant_initial(type_projet)
            contributions = self.extract_nombre_contributions(type_projet)
            montant_actuel = self.extract_montant_collecte_actuel(type_projet)
            commentaires = self.extract_nombre_commentaires()
            videos = self.extract_nombre_videos()
            images = self.extract_nombre_images()
            publications = self.extract_nombre_publications()
            
            return {
                'id': item['id'],
                'url': item['url'],
                'titre': titre,
                'adresse': adresse,
                'categorie': categorie,
                'categorie_source': item.get('categorie_source', 'inconnue'),
                'type_projet': type_projet,
                'montant_initial': montant_initial,
                'montant_actuel': montant_actuel,
                'nombre_contributions': contributions,
                'nombre_commentaires': commentaires,
                'nombre_videos': videos,
                'nombre_images': images,
                'nombre_publications': publications
            }
            
        except Exception as e:
            print(f"ERREUR Projet {item['id']}: {e}")
            # Retour par défaut si erreur
            return {
                'id': item['id'],
                'url': item['url'],
                'titre': None,
                'adresse': None,
                'categorie': None,
                'categorie_source': item.get('categorie_source', 'inconnue'),
                'type_projet': None,
                'montant_initial': None,
                'montant_actuel': None,
                'nombre_contributions': None,
                'nombre_commentaires': None,
                'nombre_videos': None,
                'nombre_images': None,
                'nombre_publications': None
            }
    
    def parcourir_projets_par_lots(self, urls_et_ids, batch_size=500):
        """Parcourt tous les projets par lots pour éviter la surcharge"""
        data_totale = []
        total_projets = len(urls_et_ids)
        print(f"DEBUT EXTRACTION: {total_projets} projets en {((total_projets-1)//batch_size)+1} lots")
        
        for batch_num in range(0, len(urls_et_ids), batch_size):
            batch = urls_et_ids[batch_num:batch_num + batch_size]
            print(f"\n--- LOT {batch_num//batch_size + 1} ---")
            print(f"Projets {batch_num+1} à {batch_num + len(batch)}")
            
            if not self.driver:
                self.init_driver()
            
            succes_lot = 0
            
            try:
                for i, item in enumerate(batch, start=1):
                    projet_data = self.extraire_donnees_projet(item)
                    data_totale.append(projet_data)
                    
                    # Affichage réduit
                    if i % 25 == 0 or i == len(batch):
                        print(f"[{batch_num + i}/{total_projets}] {item['id']} - "
                              f"Cat: {projet_data['categorie'] or 'N/A'} | "
                              f"Type: {projet_data['type_projet']} | "
                              f"Montant: {projet_data['montant_actuel'] or 'N/A'}/{projet_data['montant_initial'] or 'N/A'}")
                    
                    succes_lot += 1
                
                # Sauvegarde intermédiaire
                df_inter = pd.DataFrame(data_totale)
                df_inter.to_csv(self.partial_file, index=False, encoding='utf-8')
                print(f"SAUVEGARDE: Lot {batch_num//batch_size + 1} terminé - {succes_lot}/{len(batch)} réussis")
                
            except Exception as e:
                print(f"ERREUR LOT: {e}")
            finally:
                self.close_driver()
                time.sleep(2)  # pause entre les lots
        
        return data_totale
    
    def scraper_4_categories_france(self, max_clicks_per_category=10000, batch_size=500):
        """Scrape tous les projets des 4 catégories sélectionnées (France uniquement)"""
        print("LANCEMENT DU SCRAPING ULULE - 4 CATEGORIES SELECTIONNEES")
        print("MAX_CLICKS: 10000 par catégorie")
        print("FILTRE: Projets France uniquement (countries=FR)")
        print("=" * 70)
        print("CATEGORIES:\n- Artisanat & Cuisine\n- Mode & Design\n- Santé & Bien-être\n- Technologie")
        print("=" * 70)
        
        # Etape 1: Chargement des 4 catégories
        print("\nETAPE 1: CHARGEMENT DES URLs DES 4 CATEGORIES (FRANCE)")
        self.init_driver()
        urls_et_ids = self.load_selected_categories(max_clicks_per_category=max_clicks_per_category)
        self.close_driver()
        
        if not urls_et_ids:
            print("ERREUR: Aucune URL extraite")
            return
        
        print(f"PROJETS A TRAITER: {len(urls_et_ids)} projets uniques (France)")
        
        # Etape 2: Extraction des données
        print(f"\nETAPE 2: EXTRACTION DES DONNEES (batch_size={batch_size})")
        data_finale = self.parcourir_projets_par_lots(urls_et_ids, batch_size=batch_size)
        
        # Etape 3: Sauvegarde finale
        df_final = pd.DataFrame(data_finale)
        df_final.to_csv(self.output_file, index=False, encoding='utf-8')
        
        # Etape 4: Statistiques
        self.generer_statistiques_4_categories(df_final)
        
        print(f"\nEXTRACTION TERMINEE !")
        print(f"Fichier: {self.output_file}")
        print(f"Dataset: {len(df_final)} projets (France)")
        
        return df_final
    
    def generer_statistiques_4_categories(self, df):
        """Génère des statistiques détaillées sur les données collectées"""
        print(f"\nSTATISTIQUES DETAILLEES - 4 CATEGORIES (FRANCE)")
        print(f"Projets traités: {len(df)}")
        
        # Répartition par catégorie source
        categories_source_counts = df['categorie_source'].value_counts()
        print(f"Repartition par categorie source:")
        for categorie, count in categories_source_counts.items():
            print(f"   {categorie}: {count} projets ({(count/len(df)*100):.1f}%)")
        
        # Répartition par catégorie détectée
        categories_counts = df['categorie'].value_counts()
        print(f"Repartition par categorie detectee:")
        for categorie, count in categories_counts.items():
            print(f"   {categorie}: {count} projets ({(count/len(df)*100):.1f}%)")
        
        # Répartition par type de projet
        types_counts = df['type_projet'].value_counts()
        print(f"Types de projets:")
        for type_projet, count in types_counts.items():
            print(f"   {type_projet}: {count} projets ({(count/len(df)*100):.1f}%)")
        
        # Statistiques financières
        print(f"Statistiques financieres:")
        df_euros = df[df['type_projet'] == 'euros']
        if len(df_euros) > 0:
            print(f"   EUR - Montant moyen: {df_euros['montant_actuel'].mean():,.0f}€/{df_euros['montant_initial'].mean():,.0f}€")
        
        df_preventes = df[df['type_projet'] == 'prevente']
        if len(df_preventes) > 0:
            print(f"   PREVENTE - Ventes moyennes: {df_preventes['montant_actuel'].mean():.0f}/{df_preventes['montant_initial'].mean():.0f}")
        
        # Statistiques médias
        print(f"Statistiques medias:")
        print(f"   Projets avec videos: {(df['nombre_videos'] > 0).sum()}")
        print(f"   Projets avec images: {(df['nombre_images'] > 0).sum()}")
        print(f"   Projets avec commentaires: {(df['nombre_commentaires'] > 0).sum()}")
        print(f"   Projets avec publications: {(df['nombre_publications'] > 0).sum()}")
        
        # Moyennes médias
        print(f"MOYENNES MEDIAS:")
        print(f"   Videos: {df['nombre_videos'].mean():.1f}")
        print(f"   Images: {df['nombre_images'].mean():.1f}")
        print(f"   Commentaires: {df['nombre_commentaires'].mean():.1f}")
        print(f"   Publications: {df['nombre_publications'].mean():.1f}")
        print(f"   Contributions: {df['nombre_contributions'].mean():.1f}")
        
        print(f"Projets avec adresse: {df['adresse'].notna().sum()} ({df['adresse'].notna().sum()/len(df)*100:.1f}%)")




def main():
    """Point d'entrée principal qui gère l'exécution du scraping des 4 catégories France"""
    scraper = UluleCompleteScraper(headless=True)  # Crée une instance du scraper en mode headless
    
    try:
        print("Demarrage du scraping Ulule - 4 categories France...")
        df_resultat = scraper.scraper_4_categories_france(
            max_clicks_per_category=10000,  # Nombre max de clics par catégorie
            batch_size=500                  # Lots de 500 projets pour l'extraction
        )
        
        if df_resultat is not None:
            print("\nSCRAPING DES 4 CATEGORIES FRANCE TERMINE AVEC SUCCES!")
            print(f"Fichier sauvegardé: {scraper.output_file}")
            print(f"Nombre de projets extraits: {len(df_resultat)}")
            
    except KeyboardInterrupt:
        print("\nExtraction interrompue par l'utilisateur (Ctrl+C)")  # Gestion de l'interruption manuelle
        
    except Exception as e:
        print(f"\nErreur lors de l'extraction: {e}")  # Gestion des autres exceptions
        
    finally:
        scraper.close_driver()  # Ferme le navigateur proprement
        print("Script termine.")

# Ce code s'exécute seulement si le script est lancé directement
if __name__ == "__main__":
    main()

