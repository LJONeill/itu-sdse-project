package main

import (
	"context"

	"dagger.io/dagger"
)

//i think this file is supposed to contain the os directory making code from the notebook

func main() {

	//contains all the following functions

}

func Build(ctx context.Context) error {
	// Initialize Dagger client
	client, err := dagger.Connect(ctx)
	if err != nil {
		return err
	}
	defer client.Close()

	python := client.Container().From("python:3.12.2-bookworm").
		WithDirectory("python", client.Host().Directory("python-files")).
		WithExec([]string{"python", "--version"})

	python = python.WithExec([]string{"python", "python/hello.py"})

	_, err = python.
		Directory("output").
		Export(ctx, "output")
	if err != nil {
		return err
	}

	return nil
}
