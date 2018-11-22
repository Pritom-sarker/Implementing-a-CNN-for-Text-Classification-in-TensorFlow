import train_test
import sys
print (sys.argv)


print("Type 'train' for training \nand\n Type 'test' for testing : ")
text = input("\nyour Choice   :: ").lower()
if text=='train':
    x_train, y_train, vocab_processor, x_dev, y_dev =train_test.preprocess()
    train_test.train(x_train, y_train, vocab_processor, x_dev, y_dev)
elif text=='test':
    train_test.test()
