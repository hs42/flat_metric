library("splatter")
library("scater")

group_prob = c(0.1, 0.35, 0.1, 0.35, 0.1)
de_prob = c(0.0, 0.0, 0.3, 0.3, 0.6)
#facLoc = 

sim <- splatSimulateGroups(nGenes=5000, batchCells = 1000, de.prob = de_prob, group.prob = group_prob, verbose = FALSE)
counts <- counts(sim)
types = colData(sim)

write.csv(counts,"counts_splatter.csv")
write.csv(types,"types_splatter.csv")


sim <- logNormCounts(sim)
sim <- runPCA(sim, ncomponents = 2)

jpeg("PCAplot.jpg")
plotPCA(sim, colour_by = "Group") 
dev.off()
