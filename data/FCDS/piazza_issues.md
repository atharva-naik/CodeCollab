# Issue 1
In one case a max_vocab_len constraint needed to be added (I succeeded after adding the max_vocab_len constraint to the string slices! Thanks for your pointing out!)

# Issue 2
## Claim by user:
"On further exploration of the Cache class, I saw that the Ram Cache is empty initially however we are reading the Disk cache from a text file which I would assume would be a very large file."
## Rebuttal by instructor: 
Please take a closer look at the initialization. Both Ram and Disk caches are initially empty.
If the user noticed the "initialization" steps for Ram and Disk caches and that they were empty, they might not have made this mistake.

# Issue 3
## This person forgot to add model.train()
"Solved. It's because I didn't add model.train(). It took me really long to realise this issue. 
It seems that there will be a difference if I only call model.eval() in the training process.
Thank yo so much!"

# Issue 4
## This person had a query regarding how shuffling in DataLoader works
"Although the DataLoader contain random batches of data points when shuffle is True, the DataLoader itself doesn't change in each epoch. In other words, each epoch will train the same batches of data with the same order. Is my understanding correct? Thx!"

**This can be, and was answered through the documentation:**
"when shuffle is set to True, the DataLoader will create a new random order of samples at the start of each epoch.
You can read more : https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader"

# Issue 5: Q9 Not getting correct accuracy
## This person was getting low accuracy on the hidden test cases
Solution: "At the end of Q8 you need to load the stored state dict back into the model."
They forgot the `model.load_state_dict()` step (forgot to load the model checkpoint). I have done this mistake before too :/

# Issue 6: Timing out on sail (even though only Q1 was submitted)
## This person was getting timeout issues
Repeated list concatenation is expensive (you have to create a new list every time). You should use the appropriate methods to modify self.data and self.labels in place.