package com.deep.framework.ast.declaration;

import com.deep.framework.ast.Node;
import com.deep.framework.ast.Stream;
import com.deep.framework.ast.expression.Expression;
import com.deep.framework.ast.expression.Name;

import static com.deep.framework.ast.lexer.TokenType.AT;

public class AnnotationDeclaration extends Declaration {
    private Name annotationType;
    private Expression arguments;

    public AnnotationDeclaration(Node prarent, Name annotationType) {
        super(prarent);
        this.annotationType = annotationType;
        this.annotationType.setPrarent(this);

        getChildrens().addAll(annotationType);
    }

    public AnnotationDeclaration(Node prarent, Name annotationType, Expression arguments) {
        super(prarent);
        this.annotationType = annotationType;
        this.arguments = arguments;

        this.annotationType.setPrarent(this);
        this.arguments.setPrarent(this);

        getChildrens().addAll(annotationType, arguments);
    }

    public static void parser(Node node) {
        if (node instanceof AnnotationDeclaration) return;
        Stream.of(node.getChildrens()).reduce((list, a, b) -> {
            if (a.equals(AT) && b instanceof Name) {
                AnnotationDeclaration declare = new AnnotationDeclaration(node.getPrarent(), (Name) b);
                node.getChildrens().remove(b);
                node.replace(a, declare);
            } else if (a.equals(AT) && b instanceof CallableDeclaration c) {
                AnnotationDeclaration declare = new AnnotationDeclaration(node.getPrarent(), c.getName(), c.getParameters());
                node.getChildrens().remove(b);
                node.replace(a, declare);
            }
        });
    }

}