library(ggplot2)
library(grid)

accuracies = read.csv("results.csv")

three = accuracies[ accuracies$nr_classes == 3, ]
three_reg = three[ c("author", "reg_accuracy") ]
three_ova = three[ c("author", "ova_accuracy") ]
three_reg$method <- rep("reg", nrow(three_reg))
three_ova$method <- rep("ova", nrow(three_ova))
colnames(three_reg) = c("author", "accuracy", "method")
colnames(three_ova) = c("author", "accuracy", "method")
three = rbind(three_reg, three_ova)

four = accuracies[ accuracies$nr_classes == 4, ]
four_reg = four[ c("author", "reg_accuracy") ]
four_ova = four[ c("author", "ova_accuracy") ]
four_reg$method <- rep("reg", nrow(four_reg))
four_ova$method <- rep("ova", nrow(four_ova))
colnames(four_reg) = c("author", "accuracy", "method")
colnames(four_ova) = c("author", "accuracy", "method")
four = rbind(four_reg, four_ova)


theme = theme(
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank(),
  panel.background = element_blank(),
  panel.border = element_rect(color = "black", fill=NA, size=1),
  
  axis.title.x = element_blank(),
  axis.title.y = element_blank(),
  plot.title = element_text(hjust = 0.5),
  
  legend.position = c(.98, .98),
  legend.justification = c("right", "top"),
  legend.title = element_blank(),
  legend.key = element_rect(fill = "white", colour = NA),
  legend.box.background = element_rect(color="black", size=0.5)
)

scale_y = scale_y_continuous(limits=c(0.25, 0.8), breaks=seq(0.2, 0.85, 0.05))


three_plot = ggplot() +
  ggtitle("Average accuracies, three-class data") +
  geom_point(data = three, aes(author, accuracy, shape = method), size=4) +
  geom_rug(data = three, aes(y = accuracy), size=0.3) +
  scale_shape_manual(values=c(0, 1)) +
  scale_y +
  theme

four_plot = ggplot() +
  ggtitle("Average accuracies, three-class data") +
  geom_point(data = four, aes(author, accuracy, shape = method), size=4) +
  geom_rug(data = four, aes(y = accuracy), size=0.3) +
  scale_shape_manual(values=c(0, 1)) +
  scale_y +
  theme

grid.draw(cbind(ggplotGrob(three_plot), ggplotGrob(four_plot), size = "first"))
