package main

import (
	"context"
	"fmt"

	"dagger.io/dagger"
)

func main() {
	ctx := context.Background()

	if err := Build(ctx); err != nil {
        fmt.Println("Error:", err)
        panic(err)
    }
	//ml_pipeline()
}

func Build(ctx context.Context) error {
	// Initialize Dagger client
	client, err := dagger.Connect(ctx)
	if err != nil {
		return err
	}
	defer client.Close()

	python := client.Container().From("python:3.12.2-bookworm").
		WithDirectory("py_files", client.Host().Directory("src")).
		WithExec([]string{"python", "--version"})

	python = python.WithExec([]string{"python", "py_files/config.py"})

	_, err = python.
		Directory("output"). // Before writing to this folder, we may have to make sure it exists, 'os.makedirs('output', exist_ok=True)' 
		Export(ctx, "output") //the exist_ok=True makes nothing happen if it already exists
	if err != nil {
		return err
	}

	data := client.Container().From("python:3.12.2-bookworm").
		WithDirectory("py_files", client.Host().Directory("src")).
		WithExec([]string{"python", "py_files/dataset.py"})

	//data = data.WWithExec([]string{"python", "features.py"})

	_, err = data.
		Directory("output"). // See above re: folder creation
		Export(ctx, "output")
	if err != nil {
		return err
	}
	return nil
}