import csv

with open('/Users/arav/civai_bias/data/lexicons/loaded_vader_terms.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["token", "mean_sentiment_rating", "standard deviation"])
    blacklist=["good","great","bad","better","worse","love","hate","happy","sad","angry","pleased","dislike","enjoy","afraid","sorry"]
    with open('data/lexicons/vader_lexicon.txt', 'r') as f:
        words=[]
        lines=f.readlines()
        for line in lines:
            if '\t' in line:
                parts=line.split('\t')
                if len(parts) >= 3:
                    try:
                        sentiment = float(parts[1].strip())
                        std_dev = float(parts[2].strip())
                        if abs(sentiment) >= 1.5 and std_dev<=1.5 and len(parts[0])>2 and parts[0] not in blacklist:
                            words.append([parts[0], sentiment, std_dev])
                    except ValueError as e:
                        print(f"Error converting line: {parts[1].strip()} or {parts[2].strip()} - {e}")
            elif ' ' in line:
                parts=line.split(' ')
                if len(parts) >= 3:
                    try:
                        sentiment = float(parts[1].strip())
                        std_dev = float(parts[2].strip())
                        if abs(sentiment) >= 1.5 and std_dev<=1.5 and len(parts[0])>2 and parts[0] not in blacklist:
                            words.append([parts[0], sentiment, std_dev])
                    except ValueError as e:
                        print(f"Error converting line: {parts[1].strip()} or {parts[2].strip()} - {e}")
    writer.writerows(words)