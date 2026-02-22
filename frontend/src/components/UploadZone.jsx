import React, { useRef, useState } from 'react';
import axios from 'axios';
import { Upload } from 'lucide-react';

export default function UploadZone({ onUploadComplete }) {
    const [isDragging, setIsDragging] = useState(false);
    const fileInputRef = useRef(null);

    const handleFile = async (file) => {
        if (!file || file.type !== 'application/pdf') {
            alert('Only PDF files are supported.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await axios.post('/api/upload', formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            onUploadComplete({
                job_id: res.data.job_id,
                filename: res.data.filename,
                status: 'queued',
                progress: 0,
                message: 'Queued'
            });
        } catch (err) {
            console.error(err);
            alert('Upload failed');
        }
    };

    const onDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    };

    return (
        <div
            className={`upload-zone ${isDragging ? 'drag-active' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={(e) => { e.preventDefault(); setIsDragging(false); }}
            onDrop={onDrop}
            onClick={() => fileInputRef.current.click()}
        >
            <input
                type="file"
                accept="application/pdf"
                ref={fileInputRef}
                style={{ display: 'none' }}
                onChange={(e) => {
                    if (e.target.files.length > 0) handleFile(e.target.files[0]);
                    e.target.value = '';
                }}
            />
            <Upload size={24} className="upload-icon mx-auto mb-2" />
            <div style={{ marginTop: "10px" }}><strong>Upload Tender PDF</strong></div>
            <p>Drag and drop or click to browse</p>
        </div>
    );
}
