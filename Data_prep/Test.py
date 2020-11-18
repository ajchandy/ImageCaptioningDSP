from pickle import dump,load
with open("C:/Users/akhil/PycharmProjects/Image-captioning---DSP/pkl files/test_image_descriptions.pkl","rb") as f:
    d=load(f)



print(d['3490736665_38710f4b91'])

def remove_seq(d):
    for l in d.keys():
        caption_list = d[l]
        new_caption = []
        for c in caption_list:
            # replace special characters in the sentences
            sent = c.replace('startseq', ' ')
            sent = sent.replace('endseq', ' ')
            sent=sent.strip()
            new_caption.append(sent)

        d[l]=new_caption
    return d

dict=remove_seq(d)


def write_text_to_file(dict):
    s1,s2,s3,s4,s5=([] for i in range (5))
    for s in dict.keys():
        caption_list=dict[s]
        s1.append(caption_list[0])
        s2.append(caption_list[1])
        s3.append(caption_list[2])
        s4.append(caption_list[3])
        s5.append(caption_list[4])

    with open('C:/Users/akhil/PycharmProjects/Image-captioning---DSP/references/File1.txt', 'w') as f:
        for item in s1:
            f.write("%s\n" % item)
    with open('C:/Users/akhil/PycharmProjects/Image-captioning---DSP/references/File2.txt', 'w') as f:
        for item in s2:
            f.write("%s\n" % item)
    with open('C:/Users/akhil/PycharmProjects/Image-captioning---DSP/references/File3.txt', 'w') as f:
        for item in s3:
            f.write("%s\n" % item)
    with open('C:/Users/akhil/PycharmProjects/Image-captioning---DSP/references/File4.txt', 'w') as f:
        for item in s4:
            f.write("%s\n" % item)
    with open('C:/Users/akhil/PycharmProjects/Image-captioning---DSP/references/File5.txt', 'w') as f:
        for item in s5:
            f.write("%s\n" % item)


write_text_to_file(dict)









