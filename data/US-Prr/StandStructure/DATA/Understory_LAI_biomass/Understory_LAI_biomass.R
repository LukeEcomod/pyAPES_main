# This script creats various plots/histograms from the original data sets
# Hideki Kobayashi, Oct 6, 2022

library(ggplot2)
library(reshape2)
library(gridExtra)
library(Rmisc)

############################################################
# Understory LAI
############################################################

ulai = read.csv("Understory_LAI_biomass.csv")
ulai_sub = ulai[-17,]
ulai_odr =transform(ulai_sub, SiteID = 
                      factor(SiteID, levels = c("U-0", "U-1", "U-2", "U-3", "U-4", 
                                                "U-5", "U-6", "U-7", "U-8", "U-9",
                                                "U-10", "U-11", "U-12", "U-13", 
                                                "U-14", "U-15")))

# LAI bar plot
e1 = ggplot(ulai_odr,aes(x = SiteID, y = LAI..m2.m.2.)) +
  geom_bar(stat = "identity") +
  geom_hline(aes(yintercept=mean(ulai_odr$LAI..m2.m.2,na.rm=TRUE)),
             color="blue", linetype="dashed", size=1) +
  ylab("LAI") +
  xlab("Site ID") +
  ggtitle(paste("LAI mean =",mean(ulai_odr$LAI..m2.m.2,na.rm=TRUE),sep =" ")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1,color="black"),
        panel.background = element_rect(fill = "transparent", colour = "black",size=1),
        text = element_text(size = 24),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank())
ggsave(file = "Figures/uLAI.pdf", width = 10)
ggsave(file = "Figures/uLAI.jpg", width = 10)
show(e1)

# above ground leaf and woody biomass

cn = c("SiteID","Total_Leaf_Area","Ground.area..m.2." ,"LAI..m2.m.2.",
       "LeafBiomass","WoodyBiomass","Total_above_ground_biomass",
       "Cloudberry_leaf_biomass","Cotton_Grass_leaf_biomass",
       "Bilberry_leaf_biomass","Cranberry_leaf_biomass",
       "BogBirch_leaf_biomass", "BlackSpruce_leaf_biomass",
       "LabradorTea_leaf_biomass","Others_leaf_biomass",
       "Cloudberry_woody_biomass","Cotton_Grass_woody_biomass",
       "Bilberry_woody_biomass","Cranberry_woody_biomass","BogBirch_woody_biomass",     
       "BlackSpruce_woody_biomass","LabradorTea_woody_biomass", 
       "Others_woody_biomass","Dry_Material_biomass")
colnames(ulai_odr) = cn

# mass partitions
temp1 = melt(ulai_odr, id.vars ="SiteID") 
temp2 = subset(temp1,temp1$variable =="LeafBiomass" 
               | temp1$variable =="WoodyBiomass")
temp2$value = as.numeric(temp2$value)

# average fraction of foliage biomass fraction to woody
ff= 100*ulai$Leaf_Biomass..g...m.2.[17]/(ulai$Leaf_Biomass..g...m.2.[17]+ulai$WoodyBiomass..g.m.2.[17])

e2 = ggplot(temp2,aes(x = SiteID, y = value, fill=variable)) +
  geom_bar(stat = "identity") +
  scale_fill_brewer(palette='Set1') +
  ylab("Above ground dry mass (g)") +
  xlab("Tree ID") +
  ggtitle(paste("Mean foliage fraction =",floor(100*ff)/100,"%",sep =" ")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        panel.background = element_rect(fill = "transparent", colour = "black",size=1),
        text = element_text(size = 24),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        legend.position = c(0.8,0.9))
ggsave(file = "Figures/Understory_LAI_biomass_AboveBiomass.pdf", width = 10)
ggsave(file = "Figures/Understory_LAI_biomass_AboveBiomass.jpg", width = 10)
show(e2)

# mass partitions
# total 
Total = ulai[17,]
colnames(Total) = cn
ptemp1 = melt(Total, id.vars ="SiteID") 

# Foliage/woody portion???biomass
v= grep("leaf", ptemp1$variable ) 
ptemp2= ptemp1[v,]
ptemp2$SiteID = "Foliage"
v= grep("woody", ptemp1$variable ) 
ptemp3= ptemp1[v,]
ptemp3$SiteID = "Woody"
ptemp5 = rbind(ptemp2,ptemp3)

cn = c("Cloudberry","Cotton","Bilberry","Cranberry","BogBirch","BlackSpruce","LabradorTea","Others",
       "Cloudberry","Cotton","Bilberry","Cranberry","BogBirch","BlackSpruce","LabradorTea","Others")

ptemp5[,2] = cn

e3 = ggplot(ptemp5,aes(x = SiteID, y = value, fill=variable)) +
  geom_bar(stat = "identity", position = "fill") +
  scale_fill_brewer(palette='Set1') +
  ylab("Dry mass fraction") +
  xlab("") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        panel.background = element_rect(fill = "transparent", colour = "black",size=1),
        text = element_text(size = 24),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank())
ggsave(file = "Figures/Understory_LAI_biomass_ummary_AvobeBiomassPortion.pdf", width = 10)
ggsave(file = "Figures/Understory_LAI_biomass_Summary_AvobeBiomassPortion.jpg", width = 10)
show(e3)

# Foliage portion biomass for each Site ID
qtemp1 = melt(ulai_odr, id.vars ="SiteID") 

# Foliage/woody portion???biomass
v= grep("leaf", qtemp1$variable ) 
qtemp2= qtemp1[v,]
#qtemp2 = cbind(qtemp2,"Foliage")

cn=c(1:128)
cn[1:16] = "Cloudberry"
cn[17:32] = "Cotton"
cn[33:48] = "Bilberry"
cn[49:64] = "Cranberry"
cn[65:80] = "BogBirch"
cn[81:96] = "BlackSpruce"
cn[97:112] = "LabradorTea"
cn[113:128] = "Others"

qtemp2[,2] = cn
e4 = ggplot(qtemp2,aes(x = SiteID, y = value, fill=variable)) +
  geom_bar(stat = "identity", position = "fill") +
  scale_fill_brewer(palette='Set1') +
  ylab("Foliage dry mass") +
  xlab("") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        panel.background = element_rect(fill = "transparent", colour = "black",size=1),
        text = element_text(size = 24),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank())
ggsave(file = "Figures/Understory_LAI_biomass_FoliageBiomass.pdf", width = 10)
ggsave(file = "Figures/Understory_LAI_biomass_FoliageBiomass.jpg", width = 10)
show(e4)

# woody portion???biomass
v= grep("woody", qtemp1$variable) 
qtemp2= qtemp1[v,]

cn=c(1:128)
cn[1:16] = "Cloudberry"
cn[17:32] = "Cotton"
cn[33:48] = "Bilberry"
cn[49:64] = "Cranberry"
cn[65:80] = "BogBirch"
cn[81:96] = "BlackSpruce"
cn[97:112] = "LabradorTea"
cn[113:128] = "Others"

qtemp2[,2] = cn
e5 = ggplot(qtemp2,aes(x = SiteID, y = value, fill=variable)) +
  geom_bar(stat = "identity", position = "fill") +
  scale_fill_brewer(palette='Set1') +
  ylab("Woody dry mass") +
  xlab("") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        panel.background = element_rect(fill = "transparent", colour = "black",size=1),
        text = element_text(size = 24),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank())
ggsave(file = "Figures/Understory_LAI_biomass_WoodyBiomass.pdf", width = 10)
ggsave(file = "Figures/Understory_LAI_biomass_WoodyBiomass.jpg", width = 10)
show(e5)

## LMA summary from July 2018 
lma = read.csv("../Understory_LAI_biomass/LMA.csv")

e6 = ggplot(lma,aes(x = Species, y = LMA.g.m.2.)) +
  geom_bar(stat = "identity") +
  ylab("LMA (g / m^2)") +
  xlab(" ") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1,colour = "black"),
        panel.background = element_rect(fill = "transparent", colour = "black"),
        text = element_text(size = 24))
ggsave(file = "Figures/LMA.pdf", width = 8)
ggsave(file = "Figures/LMA.jpg", width = 8)
show(e6)


