import time 
def no_planet():
    no_planet="Based on the characteristics observed in the light curve, this signal is unlikely to be from an exoplanet. The transit shape appears sharp and V-shaped rather than flat-bottomed, which usually indicates a stellar eclipse rather than a planetary transit. The depth of the dip is also irregular compared to what would be expected from a planet, suggesting the object blocking the star is too large to be planetary in size. In addition, the odd-even transit depths show noticeable differences and the light curve displays additional brightness variations outside the main dip, both of which are typical features of an eclipsing binary system rather than a stable planetary orbit. Taken together, these indicators strongly point to the source being a stellar companion or another non-planetary phenomenon rather than a genuine exoplanet."
    
    for i in no_planet.split():
        yield i + ' '
        time.sleep(0.05)
        
def can_planet():
    can_planet="The features in this light curve point toward a plausible exoplanet candidate. The transit signal repeats consistently with a stable period, and the dip has a smooth, symmetric U-shaped profile rather than the sharper shape often seen in stellar eclipses. The depth of the transit falls within a realistic planetary range, suggesting that the object blocking the star is roughly planet-sized rather than stellar. There’s no noticeable secondary eclipse or irregular brightness variation outside the transit, and the odd and even events match closely, which further supports the idea that a single object is orbiting the star. While follow-up observations are still needed to confirm it, the evidence here aligns well with what you’d expect from a real exoplanet."
    
    for i in can_planet.split():
        yield i + ' '
        time.sleep(0.05)