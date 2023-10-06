1. MIDGNagg : thực nghiệm dòng 37 (gần giống nhân 3 nhưng trong MIDGN aggregate item trưước UB graph)
2. MIDGNall : nhân 3, thực nghiệm dòng 22 23 (có sửa 1 chút chỗ contrastive từ 6 -> 2 với mỗi thực nghiệm)
3. UBI : thực nghiệm dòng 24 25 thay UI trong CrossCBR bằng UBI 1 bản có trọng số ở layer lightGCN (25))
4. hardneg : hard negative thực nghiệm dòng 26 27 28 29(bản up lên github đã comment những đoạn code ko lấy hard negative -> tức là 100% hard negative)
5. softC : contrastive loss mà tập positive và negative có size thực nghiệm dòng 30 31 32 
7. fusionui : thử thay thế UI trong CrossCBR bằng user-item lấy từ UI và BI mà B tương tác với U (1 bản có đếm số lần (33) không đếm số lần tương tác (34))
