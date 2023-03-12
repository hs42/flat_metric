#This is a simple Splatter script, which simulates a population of 5 different groups of cell
#This script generates the raw data, which is to be analyzed in the following


library("splatter")
library("scater")

group_prob = c(0.1, 0.35, 0.1, 0.35, 0.1) #define group sizes
de_prob = c(0.0, 0.0, 0.3, 0.3, 0.6)      #define probabilities to find differentially expressed genes (-> "de") for each of these groups compared to the default parameters


sim <- splatSimulateGroups(nGenes=5000, batchCells = 1000, de.prob = de_prob, group.prob = group_prob, verbose = FALSE) #do simulation

#save counts
counts <- counts(sim) 
write.csv(counts,"counts_splatter.csv")

#save labels
types = colData(sim) 
write.csv(types,"types_splatter.csv")

#Here, we plot a preliminary PCA reduction. This is done later on more nicely in the 2--find_variable_and_plot jupyter notebook
sim <- logNormCounts(sim)
sim <- runPCA(sim, ncomponents = 2)

jpeg("PCA.jpg")
plotPCA(sim, colour_by = "Group") 
dev.off()
